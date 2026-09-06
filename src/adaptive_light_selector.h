#ifndef ADAPTIVE_LIGHT_SELECTOR_H
#define ADAPTIVE_LIGHT_SELECTOR_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "adaptive_route_model.h"

namespace adaptive_host {

// A spECK/Ocean-inspired selector for the U280 unified kernel.  The online
// policy deliberately avoids predicting kernel cycles.  Its mandatory pass
// only reads A's column indices plus the length and first/last column of each
// referenced B row.  A small, conditional HLL sample is used only for rows
// whose product/span lower bound cannot make a confident decision.
struct LightSelectorConfig {
	uint32_t dense_capacity = 65536;
	// Legacy kernels address DENSE with a matrix-global column index and must
	// therefore reject DENSE when N exceeds dense_capacity.  The row-persistent
	// kernel instead subtracts a per-fragment dense_base before the physical
	// workspace.  In that design a complete row remains the selector unit:
	// rows whose span fits use one fragment and wider, high-reuse rows may use
	// several disjoint column fragments without changing their selected mode.
	bool row_local_dense = false;
	bool allow_segmented_dense = false;
	uint32_t dense_min_runs = 17;
	uint64_t dense_min_products = 512;
	double dense_min_compression = 8.0;
	double sampled_dense_margin = 1.20;
	double sampled_min_lower_bound = 0.50;
	double sample_rate = 0.03;
	uint32_t max_sample_rows = 512;
	uint32_t sample_min_avg_products = 64;
	uint32_t min_dense_batch_rows = 2;
	uint64_t dense_single_row_products = 8192;
	double min_adaptive_dense_work = 0.10;
	double dense_only_work = 0.90;
	bool enable_sampling = true;
};

struct LightSelectorReport {
	uint64_t cheap_edges = 0;
	uint64_t hll_build_entries = 0;
	uint64_t sampled_products = 0;
	uint32_t eligible_rows = 0;
	uint32_t ambiguous_rows = 0;
	uint32_t sampled_rows = 0;
	double sampled_compression = 0.0;
	double dense_work_fraction = 0.0;
	bool used_sampling = false;
};

struct LightRowFeature {
	uint32_t runs = 0;
	uint64_t products = 0;
	uint32_t max_run = 0;
	uint32_t min_column = 0;
	uint32_t max_column = 0;
	uint32_t span = 0;
	uint8_t bin = 0;
	double compression_lower_bound = 0.0;
};

namespace light_selector_detail {

constexpr unsigned kProductBins = 8;
constexpr unsigned kReuseBins = 6;
constexpr unsigned kBins = kProductBins * kReuseBins;

inline unsigned floor_log2_u64(uint64_t value) {
	unsigned result = 0;
	while (value > 1) {
		value >>= 1;
		++result;
	}
	return result;
}

inline uint8_t feature_bin(uint64_t products, double lower_bound) {
	const unsigned product_bin = std::min<unsigned>(
		kProductBins - 1, floor_log2_u64(std::max<uint64_t>(1, products)) / 3);
	unsigned reuse_bin = 0;
	if (lower_bound >= 8.0) reuse_bin = 5;
	else if (lower_bound >= 4.0) reuse_bin = 4;
	else if (lower_bound >= 2.0) reuse_bin = 3;
	else if (lower_bound >= 1.0) reuse_bin = 2;
	else if (lower_bound >= 0.5) reuse_bin = 1;
	return static_cast<uint8_t>(product_bin * kReuseBins + reuse_bin);
}

inline double median(std::vector<double> values) {
	if (values.empty()) return 0.0;
	const size_t middle = values.size() / 2;
	std::nth_element(values.begin(), values.begin() + middle, values.end());
	const double upper = values[middle];
	if ((values.size() & 1U) != 0) return upper;
	std::nth_element(values.begin(), values.begin() + middle - 1, values.end());
	return 0.5 * (values[middle - 1] + upper);
}

}  // namespace light_selector_detail

inline Plan choose_light_plan(const std::vector<uint32_t>& rowptr,
		const std::vector<uint32_t>& columns, uint32_t column_count,
		const LightSelectorConfig& config = LightSelectorConfig(),
		LightSelectorReport* report_out = nullptr) {
	using namespace light_selector_detail;
	const uint32_t row_count = static_cast<uint32_t>(rowptr.size() - 1);
	std::vector<LightRowFeature> features(row_count);
	LightSelectorReport report;
	uint64_t total_multi_products = 0;

	// Mandatory spECK-style pass: O(nnz(A)); no B-row interior is inspected.
	for (uint32_t row = 0; row < row_count; ++row) {
		LightRowFeature& feature = features[row];
		feature.runs = rowptr[row + 1] - rowptr[row];
		uint32_t min_column = column_count;
		uint32_t max_column = 0;
		bool has_product = false;
		for (uint32_t pos = rowptr[row]; pos < rowptr[row + 1]; ++pos) {
			const uint32_t b_row = columns[pos];
			const uint32_t begin = rowptr[b_row];
			const uint32_t end = rowptr[b_row + 1];
			const uint32_t length = end - begin;
			feature.products += length;
			feature.max_run = std::max(feature.max_run, length);
			if (length != 0) {
				min_column = std::min(min_column, columns[begin]);
				max_column = std::max(max_column, columns[end - 1]);
				has_product = true;
			}
			++report.cheap_edges;
		}
		if (has_product) {
			feature.min_column = min_column;
			feature.max_column = max_column;
			feature.span = max_column - min_column + 1;
			feature.compression_lower_bound = static_cast<double>(feature.products)
				/ std::max<uint32_t>(1, feature.span);
			feature.bin = feature_bin(
				feature.products, feature.compression_lower_bound);
		}
		if (feature.runs > 1) total_multi_products += feature.products;
	}

	std::array<std::vector<uint32_t>, kBins> ambiguous_by_bin;
	for (uint32_t row = 0; row < row_count; ++row) {
		const LightRowFeature& feature = features[row];
		const bool dense_legal = config.allow_segmented_dense
			|| (config.row_local_dense
				? feature.span <= config.dense_capacity
				: column_count <= config.dense_capacity);
		if (!dense_legal || feature.runs < config.dense_min_runs
				|| feature.products < config.dense_min_products)
			continue;
		++report.eligible_rows;
		if (feature.compression_lower_bound < config.dense_min_compression) {
			ambiguous_by_bin[feature.bin].push_back(row);
			++report.ambiguous_rows;
		}
	}

	std::array<double, kBins> sampled_bin_compression{};
	const double average_products = row_count == 0 ? 0.0
		: static_cast<double>(total_multi_products) / row_count;
	if (config.enable_sampling && report.ambiguous_rows != 0
			&& average_products >= config.sample_min_avg_products
			&& config.max_sample_rows != 0) {
		// Ocean builds compact B-row sketches once, but unlike the old detailed
		// selector we merge them for only a bounded, stratified row sample.
		std::vector<QuickHll> b_hll(row_count);
		for (uint32_t row = 0; row < row_count; ++row)
			for (uint32_t pos = rowptr[row]; pos < rowptr[row + 1]; ++pos) {
				quick_hll_add(b_hll[row], columns[pos]);
				++report.hll_build_entries;
			}

		uint32_t samples_left = config.max_sample_rows;
		for (unsigned bin = 0; bin < kBins && samples_left != 0; ++bin) {
			const std::vector<uint32_t>& rows = ambiguous_by_bin[bin];
			if (rows.empty()) continue;
			uint32_t wanted = static_cast<uint32_t>(
				std::ceil(config.sample_rate * rows.size()));
			wanted = std::max<uint32_t>(1, wanted);
			wanted = std::min<uint32_t>(wanted,
				std::min<uint32_t>(samples_left, rows.size()));
			std::vector<double> compression;
			compression.reserve(wanted);
			for (uint32_t sample = 0; sample < wanted; ++sample) {
				const size_t at = static_cast<size_t>(sample) * rows.size() / wanted;
				const uint32_t row = rows[at];
				const LightRowFeature& feature = features[row];
				QuickHll union_hll{{}};
				for (uint32_t pos = rowptr[row]; pos < rowptr[row + 1]; ++pos) {
					const QuickHll& sketch = b_hll[columns[pos]];
					for (unsigned reg = 0; reg < kQuickHllRegisters; ++reg)
						union_hll[reg] = std::max(union_hll[reg], sketch[reg]);
				}
				const uint32_t unique = quick_hll_estimate(union_hll,
					std::min<uint64_t>(feature.products, column_count));
				const double ratio = static_cast<double>(feature.products)
					/ std::max<uint32_t>(1, unique);
				compression.push_back(ratio);
				report.sampled_products += feature.products;
				report.sampled_compression += ratio;
				++report.sampled_rows;
			}
			samples_left -= wanted;
			sampled_bin_compression[bin] = median(std::move(compression));
		}
		report.used_sampling = report.sampled_rows != 0;
		if (report.sampled_rows != 0)
			report.sampled_compression /= report.sampled_rows;
	}

	Plan plan;
	plan.route.resize(row_count, 0);
	for (uint32_t row = 0; row < row_count; ++row) {
		const LightRowFeature& feature = features[row];
		if (feature.products == 0 || feature.runs == 0) {
			plan.route[row] = 0;
			++plan.empty_rows;
			continue;
		}
		if (feature.runs == 1) {
			plan.route[row] = 1;
			++plan.direct_rows;
			continue;
		}
		++plan.multi_rows;
		plan.multi_products += feature.products;
		const double sampled_compression = sampled_bin_compression[feature.bin];
		const bool confident_lower_bound =
			feature.compression_lower_bound >= config.dense_min_compression;
		const bool sampled_dense = sampled_compression
				>= config.dense_min_compression * config.sampled_dense_margin
			&& feature.compression_lower_bound >= config.sampled_min_lower_bound;
		const bool dense_legal = config.allow_segmented_dense
			|| (config.row_local_dense
				? feature.span <= config.dense_capacity
				: column_count <= config.dense_capacity);
		const bool use_dense = dense_legal
			&& feature.runs >= config.dense_min_runs
			&& feature.products >= config.dense_min_products
			&& (confident_lower_bound || sampled_dense);
		plan.route[row] = use_dense ? 3 : 2;
		if (use_dense) {
			++plan.dense_rows;
			plan.dense_products += feature.products;
		} else {
			++plan.merge_rows;
			plan.merge_products += feature.products;
		}
	}

	// The deployed U280 dispatcher amortizes DENSE setup only for the fixed
	// aligned contexts (0,1), (2,3), ... .  A merely consecutive candidate run
	// such as (1,2) is not a hardware batch: both rows execute as single-row
	// reductions.  Gate each unpaired candidate independently unless it carries
	// enough products to amortize a DENSE invocation on its own.
	if (config.min_dense_batch_rows > 1) {
		for (uint32_t row = 0; row < row_count; ++row) {
			if (plan.route[row] != 3) continue;
			const uint32_t partner = row ^ 1U;
			const bool aligned_pair = partner < row_count
				&& plan.route[partner] == 3;
			if (!aligned_pair
					&& features[row].products < config.dense_single_row_products) {
				plan.route[row] = 2;
				--plan.dense_rows;
				++plan.merge_rows;
				plan.dense_products -= features[row].products;
				plan.merge_products += features[row].products;
			}
		}
	}

	const uint64_t routed_products = plan.merge_products + plan.dense_products;
	report.dense_work_fraction = routed_products == 0 ? 0.0
		: static_cast<double>(plan.dense_products) / routed_products;
	if (plan.dense_rows == 0) {
		plan.mode = plan.model_mode = MERGE_ONLY;
		plan.decision_reason = "light-no-confident-dense-region";
	} else if (plan.merge_rows == 0
			|| report.dense_work_fraction >= config.dense_only_work) {
		plan.mode = plan.model_mode = DENSE_ONLY;
		plan.decision_reason = "light-dense-work-dominant";
	} else if (report.dense_work_fraction >= config.min_adaptive_dense_work) {
		plan.mode = plan.model_mode = ADAPTIVE;
		plan.decision_reason = "light-two-regimes-have-material-work";
	} else {
		plan.mode = plan.model_mode = MERGE_ONLY;
		plan.decision_reason = "light-dense-region-too-small";
	}
	if (report_out != nullptr) *report_out = report;
	return plan;
}

}  // namespace adaptive_host

#endif  // ADAPTIVE_LIGHT_SELECTOR_H
