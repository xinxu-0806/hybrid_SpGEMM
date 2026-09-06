#ifndef ADAPTIVE_ROUTE_MODEL_H
#define ADAPTIVE_ROUTE_MODEL_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace adaptive_host {

constexpr unsigned kMergeWidth = 16;
constexpr unsigned kWideMergeMaxRun = 2048;
// The TAPA DENSE datapath has eight physical banks and uses the same XOR-fold
// mapping as tapa_dense_bank().  Selector features must describe that physical
// router; the former four-bank low-bit model substantially overpredicted DENSE.
constexpr unsigned kDenseBanks = 8;
constexpr unsigned kDenseHexCapacity = 8192;
constexpr unsigned kDenseOctCapacity = 16384;
constexpr unsigned kDenseQuadCapacity = 32768;
constexpr unsigned kHllRegisters = 32;
constexpr unsigned kQuickHllRegisters = 16;

enum GlobalMode { MERGE_ONLY, DENSE_ONLY, ADAPTIVE };

using Hll = std::array<uint8_t, kHllRegisters>;

struct BankSummary {
	uint32_t length = 0;
	std::array<uint32_t, kDenseBanks> load{{}};
};

struct BRowSummary {
	uint32_t length = 0;
	uint32_t max_col = 0;
	Hll hll{{}};
	BankSummary bank;
};

struct RowFeatures {
	uint32_t runs = 0;
	uint64_t products = 0;
	uint32_t max_run = 0;
	uint32_t max_col = 0;
	uint32_t unique_columns = 0;
	uint64_t bank_replays = 0;
	uint64_t merge_materialized = 0;
	uint64_t merge_critical = 0;
	uint64_t merge_transactions = 0;
};

using QuickHll = std::array<uint8_t, kQuickHllRegisters>;

struct QuickBRowSummary {
	uint32_t length = 0;
	QuickHll hll{{}};
};

struct QuickHllState {
	QuickHll registers{{}};
	uint32_t estimate = 0;
	uint64_t upper = 0;
};

// A selector profile belongs to one board-tested datapath.  When the dense
// batching or merge implementation changes, its calibration must change with
// it; a selector must never silently reuse coefficients from an older xclbin.
struct CostModel {
	double producer_per_product = 1.0;
	double output_per_nnz = 1.0;
	double merge_per_product = 0.0;
	double merge_max_run = 0.0;
	double merge_critical = 0.0;
	double merge_materialized = 1.0;
	double merge_final_copy = 1.0;
	double merge_transaction = 32.0;
	double dense_per_product = 0.0;
	double dense_bank_replay = 0.0;
	double dense_round_ii = 6.0;
	double dense_extract_per_nnz = 1.0;
	double dense_row_setup = 48.0;
	// Aligned 2/4/8/16-row batches share one dense-core invocation.  Credits are
	// the total saved setup cost and are applied greedily in the same order and
	// at the same column-capacity gates as the unified RTL dispatcher.
	double dense_pair_setup_credit = 0.0;
	double dense_quad_setup_credit = 0.0;
	double dense_oct_setup_credit = 0.0;
	double dense_hex_setup_credit = 0.0;
	double adaptive_row_control = 2.0;
	double row_dense_margin = 1.10;
	double global_adaptive_margin = 1.03;
	uint32_t dense_min_runs = 2;
	double dense_short_row_penalty = 1.0;
};

struct Plan {
	GlobalMode mode = MERGE_ONLY;
	GlobalMode model_mode = MERGE_ONLY;
	std::vector<uint32_t> route;
	double merge_only_cycles = 0.0;
	double dense_only_cycles = std::numeric_limits<double>::infinity();
	double adaptive_cycles = std::numeric_limits<double>::infinity();
	double dense_only_product_cycles = 0.0;
	double dense_only_route_cycles = 0.0;
	double dense_only_extract_cycles = 0.0;
	double dense_only_setup_cycles = 0.0;
	double dense_only_batch_credit_cycles = 0.0;
	double predicted_adaptive_gain_cycles = 0.0;
	double selector_overhead_ms = 0.0;
	double predicted_gain_ms = 0.0;
	uint64_t summary_work = 0;
	uint32_t merge_rows = 0;
	uint32_t dense_rows = 0;
	uint32_t dense_only_pairs = 0;
	uint32_t adaptive_dense_pairs = 0;
	uint32_t dense_only_quads = 0;
	uint32_t adaptive_dense_quads = 0;
	uint32_t dense_only_octs = 0;
	uint32_t adaptive_dense_octs = 0;
	uint32_t dense_only_hexes = 0;
	uint32_t adaptive_dense_hexes = 0;
	uint32_t multi_rows = 0;
	uint64_t multi_products = 0;
	uint32_t wide_rows = 0;
	uint64_t wide_products = 0;
	uint32_t direct_rows = 0;
	uint32_t empty_rows = 0;
	uint64_t merge_products = 0;
	uint64_t dense_products = 0;
	double active_product_cv = 0.0;
	const char* decision_reason = "best-only";
};

inline uint32_t hash32(uint32_t value) {
	uint32_t x = value + 0x9e3779b9U;
	x = (x ^ (x >> 16)) * 0x85ebca6bU;
	x = (x ^ (x >> 13)) * 0xc2b2ae35U;
	return x ^ (x >> 16);
}

inline uint8_t hll_rank(uint32_t remainder) {
	if (remainder == 0) return 31;
#if defined(__GNUC__)
	const unsigned bits = 32U - static_cast<unsigned>(__builtin_clz(remainder));
#else
	unsigned bits = 0;
	for (uint32_t x = remainder; x != 0; x >>= 1) ++bits;
#endif
	return static_cast<uint8_t>(std::min(31U, 28U - std::min(28U, bits)));
}

inline uint8_t quick_hll_rank(uint32_t remainder) {
	if (remainder == 0) return 31;
#if defined(__GNUC__)
	const unsigned bits = 32U - static_cast<unsigned>(__builtin_clz(remainder));
#else
	unsigned bits = 0;
	for (uint32_t x = remainder; x != 0; x >>= 1) ++bits;
#endif
	return static_cast<uint8_t>(std::min(31U, 29U - std::min(29U, bits)));
}

inline void quick_hll_add(QuickHll& registers, uint32_t column) {
	const uint32_t hashed = hash32(column);
	const unsigned index = hashed & (kQuickHllRegisters - 1);
	registers[index] = std::max(registers[index], quick_hll_rank(hashed >> 4));
}

inline uint32_t quick_hll_estimate(
		const QuickHll& registers, uint64_t upper_bound) {
	if (upper_bound == 0) return 0;
	double inverse_sum = 0.0;
	unsigned zeros = 0;
	for (uint8_t value : registers) {
		inverse_sum += std::ldexp(1.0, -static_cast<int>(value));
		if (value == 0) ++zeros;
	}
	double estimate = 0.673 * kQuickHllRegisters * kQuickHllRegisters
		/ inverse_sum;
	if (estimate <= 2.5 * kQuickHllRegisters && zeros != 0)
		estimate = kQuickHllRegisters
			* std::log(static_cast<double>(kQuickHllRegisters) / zeros);
	const uint64_t rounded = static_cast<uint64_t>(std::llround(estimate));
	return static_cast<uint32_t>(std::max<uint64_t>(
		1, std::min<uint64_t>(upper_bound, rounded)));
}

inline void hll_add(Hll& registers, uint32_t column) {
	const uint32_t hashed = hash32(column);
	const unsigned index = hashed & (kHllRegisters - 1);
	registers[index] = std::max(registers[index], hll_rank(hashed >> 5));
}

inline uint32_t hll_estimate(const Hll& registers, uint64_t upper_bound) {
	if (upper_bound == 0) return 0;
	double inverse_sum = 0.0;
	unsigned zeros = 0;
	for (uint8_t value : registers) {
		inverse_sum += std::ldexp(1.0, -static_cast<int>(value));
		if (value == 0) ++zeros;
	}
	double estimate = 0.697 * kHllRegisters * kHllRegisters / inverse_sum;
	if (estimate <= 2.5 * kHllRegisters && zeros != 0)
		estimate = kHllRegisters
			* std::log(static_cast<double>(kHllRegisters) / zeros);
	const uint64_t rounded = static_cast<uint64_t>(std::llround(estimate));
	return static_cast<uint32_t>(std::max<uint64_t>(
		1, std::min<uint64_t>(upper_bound, rounded)));
}

inline uint8_t dense_bank(uint32_t column) {
	return static_cast<uint8_t>(
		(column ^ (column >> 3) ^ (column >> 6)) & (kDenseBanks - 1));
}

inline BankSummary make_bank_summary(
		const std::vector<uint32_t>& columns, uint32_t begin, uint32_t end) {
	BankSummary result;
	result.length = end - begin;
	for (uint32_t pos = begin; pos < end; ++pos)
		++result.load[dense_bank(columns[pos])];
	return result;
}

struct HllState {
	Hll registers{{}};
	uint32_t estimate = 0;
	uint64_t upper = 0;
};

inline HllState union_state(const HllState& left, const HllState& right,
		uint32_t column_count) {
	HllState result;
	for (unsigned i = 0; i < kHllRegisters; ++i)
		result.registers[i] = std::max(left.registers[i], right.registers[i]);
	result.upper = left.upper + right.upper;
	result.estimate = hll_estimate(result.registers,
		std::min<uint64_t>(result.upper, column_count));
	return result;
}

inline HllState reduce_hll_states(std::vector<HllState> states,
		uint32_t column_count, uint64_t& critical) {
	critical = 0;
	while (states.size() > 1) {
		std::vector<HllState> next;
		next.reserve((states.size() + 1) / 2);
		uint32_t level_max = 0;
		for (size_t i = 0; i < states.size(); i += 2) {
			HllState output = i + 1 == states.size()
				? states[i] : union_state(states[i], states[i + 1], column_count);
			level_max = std::max(level_max, output.estimate);
			next.push_back(output);
		}
		critical += level_max;
		states.swap(next);
	}
	return states.empty() ? HllState() : states.front();
}

inline void fixed_width_merge(std::vector<HllState> stage,
		uint32_t column_count, uint64_t& materialized,
		uint64_t& critical, uint64_t& transactions) {
	materialized = critical = transactions = 0;
	while (stage.size() > 1) {
		std::vector<HllState> next;
		next.reserve((stage.size() + kMergeWidth - 1) / kMergeWidth);
		uint32_t level_max = 0;
		for (size_t base = 0; base < stage.size(); base += kMergeWidth) {
			const size_t end = std::min(stage.size(), base + kMergeWidth);
			HllState output = stage[base];
			for (size_t index = base + 1; index < end; ++index) {
				output.upper += stage[index].upper;
				for (unsigned reg = 0; reg < kHllRegisters; ++reg)
					output.registers[reg] = std::max(
						output.registers[reg], stage[index].registers[reg]);
			}
			output.estimate = hll_estimate(output.registers,
				std::min<uint64_t>(output.upper, column_count));
			materialized += output.estimate;
			level_max = std::max(level_max, output.estimate);
			++transactions;
			next.push_back(output);
		}
		critical += level_max;
		stage.swap(next);
	}
}

// Stage 1 uses half-width HLL summaries and deliberately omits bank replay.
// Omitting replay makes dense look better, so a confident merge-only result is
// a conservative fast exit.  Ambiguous inputs proceed to the 32-register and
// exact replay-summary stage below.
inline std::vector<RowFeatures> extract_quick_features(
		const std::vector<uint32_t>& rowptr,
		const std::vector<uint32_t>& columns, uint32_t column_count,
		uint64_t* summary_work = nullptr) {
	const uint32_t rows = static_cast<uint32_t>(rowptr.size() - 1);
	std::vector<QuickBRowSummary> summaries(rows);
	uint64_t work = columns.size();
	for (uint32_t row = 0; row < rows; ++row) {
		QuickBRowSummary& summary = summaries[row];
		summary.length = rowptr[row + 1] - rowptr[row];
		for (uint32_t pos = rowptr[row]; pos < rowptr[row + 1]; ++pos)
			quick_hll_add(summary.hll, columns[pos]);
	}

	std::vector<RowFeatures> features(rows);
	for (uint32_t row = 0; row < rows; ++row) {
		RowFeatures& out = features[row];
		out.runs = rowptr[row + 1] - rowptr[row];
		QuickHll union_hll{{}};
		std::vector<QuickHllState> stage;
		stage.reserve(out.runs);
		for (uint32_t pos = rowptr[row]; pos < rowptr[row + 1]; ++pos) {
			const QuickBRowSummary& summary = summaries[columns[pos]];
			out.products += summary.length;
			out.max_run = std::max(out.max_run, summary.length);
			QuickHllState leaf;
			leaf.registers = summary.hll;
			leaf.estimate = summary.length;
			leaf.upper = summary.length;
			stage.push_back(leaf);
			for (unsigned reg = 0; reg < kQuickHllRegisters; ++reg)
				union_hll[reg] = std::max(union_hll[reg], summary.hll[reg]);
			work += kQuickHllRegisters;
		}
		out.unique_columns = quick_hll_estimate(union_hll,
			std::min<uint64_t>(out.products, column_count));
		while (stage.size() > 1) {
			std::vector<QuickHllState> next;
			next.reserve((stage.size() + kMergeWidth - 1) / kMergeWidth);
			for (size_t base = 0; base < stage.size(); base += kMergeWidth) {
				const size_t end = std::min(stage.size(), base + kMergeWidth);
				QuickHllState output = stage[base];
				for (size_t index = base + 1; index < end; ++index) {
					output.upper += stage[index].upper;
					for (unsigned reg = 0; reg < kQuickHllRegisters; ++reg)
						output.registers[reg] = std::max(
							output.registers[reg], stage[index].registers[reg]);
				}
				output.estimate = quick_hll_estimate(output.registers,
					std::min<uint64_t>(output.upper, column_count));
				out.merge_materialized += output.estimate;
				++out.merge_transactions;
				next.push_back(output);
			}
			stage.swap(next);
		}
	}
	if (summary_work != nullptr) *summary_work = work;
	return features;
}

// Build all online features without materializing a single partial product.
// Work is O(nnz(B) + 32*nnz(A)) for square A=B, rather than O(products).
inline std::vector<RowFeatures> extract_matrix_features(
		const std::vector<uint32_t>& rowptr,
		const std::vector<uint32_t>& columns, uint32_t column_count,
		uint64_t* summary_work = nullptr) {
	const uint32_t rows = static_cast<uint32_t>(rowptr.size() - 1);
	std::vector<BRowSummary> summaries(rows);
	uint64_t work = columns.size();
	for (uint32_t row = 0; row < rows; ++row) {
		BRowSummary& summary = summaries[row];
		summary.length = rowptr[row + 1] - rowptr[row];
		for (uint32_t pos = rowptr[row]; pos < rowptr[row + 1]; ++pos)
			hll_add(summary.hll, columns[pos]);
		if (summary.length != 0) summary.max_col = columns[rowptr[row + 1] - 1];
		summary.bank = make_bank_summary(columns, rowptr[row], rowptr[row + 1]);
	}

	std::vector<RowFeatures> features(rows);
	for (uint32_t row = 0; row < rows; ++row) {
		RowFeatures& out = features[row];
		out.runs = rowptr[row + 1] - rowptr[row];
		Hll union_hll{{}};
		std::vector<HllState> leaves;
		leaves.reserve(out.runs);
		std::array<uint64_t, kDenseBanks> bank_load{{}};
		for (uint32_t pos = rowptr[row]; pos < rowptr[row + 1]; ++pos) {
			const BRowSummary& summary = summaries[columns[pos]];
			out.products += summary.length;
			out.max_run = std::max(out.max_run, summary.length);
			out.max_col = std::max(out.max_col, summary.max_col);
			for (unsigned reg = 0; reg < kHllRegisters; ++reg)
				union_hll[reg] = std::max(union_hll[reg], summary.hll[reg]);
			HllState leaf;
			leaf.registers = summary.hll;
			leaf.estimate = summary.length;
			leaf.upper = summary.length;
			leaves.push_back(leaf);
			for (unsigned bank = 0; bank < kDenseBanks; ++bank)
				bank_load[bank] += summary.bank.load[bank];
			work += kHllRegisters;
		}
		// With eight simultaneously visible shard packet heads, packet-local
		// conflicts can be hidden by work from another source.  The unavoidable
		// replay lower bound is therefore the hottest physical bank minus the
		// ideal P/8 route length, not the sum of old four-lane packet conflicts.
		const uint64_t hottest_bank =
			*std::max_element(bank_load.begin(), bank_load.end());
		const uint64_t ideal_beats =
			(out.products + kDenseBanks - 1) / kDenseBanks;
		out.bank_replays = hottest_bank > ideal_beats
			? hottest_bank - ideal_beats : 0;
		out.unique_columns = hll_estimate(union_hll,
			std::min<uint64_t>(out.products, column_count));
		if (out.runs > 1)
			fixed_width_merge(leaves, column_count, out.merge_materialized,
				out.merge_critical, out.merge_transactions);
	}
	if (summary_work != nullptr) *summary_work = work;
	return features;
}

inline double merge_reduce_cost(const RowFeatures& row, const CostModel& model) {
	if (row.runs <= 1) return 0.0;
	return model.merge_per_product * row.products
		+ model.merge_max_run * row.max_run
		+ model.merge_critical * row.merge_critical
		+ model.merge_materialized * row.merge_materialized
		+ model.merge_final_copy * row.unique_columns
		+ model.merge_transaction * row.merge_transactions;
}

inline double dense_reduce_cost(const RowFeatures& row, const CostModel& model) {
	if (row.runs <= 1) return 0.0;
	const uint64_t groups =
		(row.products + kDenseBanks - 1) / kDenseBanks;
	return model.dense_row_setup
		+ model.dense_per_product * row.products
		+ model.dense_round_ii * (groups + row.bank_replays)
		+ model.dense_bank_replay * row.bank_replays
		+ model.dense_extract_per_nnz * row.unique_columns;
}

inline Plan choose_plan(const std::vector<RowFeatures>& rows, uint32_t N,
		const CostModel& model = CostModel(), uint32_t dense_capacity = 65536,
		uint64_t summary_work = 0) {
	Plan plan;
	plan.summary_work = summary_work;
	plan.route.resize(rows.size(), 0);
	const bool dense_legal = N <= dense_capacity;
	if (dense_legal) plan.dense_only_cycles = 0.0;
	plan.adaptive_cycles = dense_legal ? 0.0
		: std::numeric_limits<double>::infinity();
	double active_product_sum = 0.0;
	double active_product_square_sum = 0.0;
	uint32_t active_rows = 0;
	for (size_t i = 0; i < rows.size(); ++i) {
		const RowFeatures& row = rows[i];
		const double common = model.producer_per_product * row.products
			+ model.output_per_nnz * row.unique_columns;
		if (row.products == 0 || row.runs == 0) {
			plan.route[i] = 0;
			++plan.empty_rows;
			continue;
		}
		active_product_sum += row.products;
		active_product_square_sum += static_cast<double>(row.products)
			* row.products;
		++active_rows;
		if (row.runs == 1) {
			plan.route[i] = 1;
			++plan.direct_rows;
			plan.merge_only_cycles += common;
			if (dense_legal) {
				plan.dense_only_cycles += common;
				plan.adaptive_cycles += common;
			}
			continue;
		}
		const double merge = merge_reduce_cost(row, model);
		const double dense_physical = dense_reduce_cost(row, model);
		++plan.multi_rows;
		plan.multi_products += row.products;
		if (row.runs <= kMergeWidth && row.max_run <= kWideMergeMaxRun) {
			++plan.wide_rows;
			plan.wide_products += row.products;
		}
		double dense_for_route = dense_physical;
		if (row.runs < model.dense_min_runs)
			dense_for_route = std::max(dense_for_route,
				merge * model.dense_short_row_penalty);
		plan.merge_only_cycles += common + merge;
		if (dense_legal) {
			// A row-routing policy may deliberately reject short dense rows,
			// but FORCE_DENSE still executes them in the real dense datapath.
			// Do not contaminate the global dense-only prediction with the
			// selector's short-row policy penalty.
			plan.dense_only_cycles += common + dense_physical;
			const uint64_t dense_groups =
				(row.products + kDenseBanks - 1) / kDenseBanks;
			plan.dense_only_product_cycles
				+= model.dense_per_product * row.products;
			plan.dense_only_route_cycles += model.dense_round_ii
					* (dense_groups + row.bank_replays)
				+ model.dense_bank_replay * row.bank_replays;
			plan.dense_only_extract_cycles
				+= model.dense_extract_per_nnz * row.unique_columns;
			plan.dense_only_setup_cycles += model.dense_row_setup;
			const bool use_dense = row.runs >= model.dense_min_runs
				&& dense_for_route * model.row_dense_margin < merge;
			plan.adaptive_cycles += common
				+ (use_dense ? dense_for_route : merge)
				+ model.adaptive_row_control;
			plan.route[i] = use_dense ? 3 : 2;
			if (use_dense) ++plan.dense_rows;
			else ++plan.merge_rows;
			if (use_dense) plan.dense_products += row.products;
			else plan.merge_products += row.products;
		} else {
			plan.route[i] = 2;
			++plan.merge_rows;
		}
	}
	if (dense_legal) {
		auto apply_batches = [&](bool adaptive) {
			double& cycles = adaptive
				? plan.adaptive_cycles : plan.dense_only_cycles;
			uint32_t& pairs = adaptive
				? plan.adaptive_dense_pairs : plan.dense_only_pairs;
			uint32_t& quads = adaptive
				? plan.adaptive_dense_quads : plan.dense_only_quads;
			uint32_t& octs = adaptive
				? plan.adaptive_dense_octs : plan.dense_only_octs;
			uint32_t& hexes = adaptive
				? plan.adaptive_dense_hexes : plan.dense_only_hexes;
			double ignored_adaptive_credit = 0.0;
			double& credited = adaptive ? ignored_adaptive_credit
				: plan.dense_only_batch_credit_cycles;
			auto eligible = [&](size_t row) {
				return adaptive
					? plan.route[row] == 3
					: rows[row].products != 0 && rows[row].runs > 1;
			};
			auto batch_eligible = [&](size_t row, size_t batch) {
				if ((row & (batch - 1)) != 0 || row + batch > rows.size())
					return false;
				for (size_t offset = 0; offset < batch; ++offset)
					if (!eligible(row + offset)) return false;
				return true;
			};
			for (size_t row = 0; row < rows.size();) {
				if (N <= kDenseHexCapacity && batch_eligible(row, 16)) {
					cycles = std::max(0.0,
						cycles - model.dense_hex_setup_credit);
					credited += model.dense_hex_setup_credit;
					++hexes;
					row += 16;
					continue;
				}
				if (N <= kDenseOctCapacity && batch_eligible(row, 8)) {
					cycles = std::max(0.0,
						cycles - model.dense_oct_setup_credit);
					credited += model.dense_oct_setup_credit;
					++octs;
					row += 8;
					continue;
				}
				if (N <= kDenseQuadCapacity && batch_eligible(row, 4)) {
					cycles = std::max(0.0,
						cycles - model.dense_quad_setup_credit);
					credited += model.dense_quad_setup_credit;
					++quads;
					row += 4;
					continue;
				}
				if (batch_eligible(row, 2)) {
					cycles = std::max(0.0,
						cycles - model.dense_pair_setup_credit);
					credited += model.dense_pair_setup_credit;
					++pairs;
					row += 2;
					continue;
				}
				++row;
			}
		};
		apply_batches(false);
		apply_batches(true);
	}
	if (active_rows != 0 && active_product_sum != 0.0) {
		const double mean = active_product_sum / active_rows;
		const double variance = std::max(0.0,
			active_product_square_sum / active_rows - mean * mean);
		plan.active_product_cv = std::sqrt(variance) / mean;
	}

	plan.mode = plan.dense_only_cycles < plan.merge_only_cycles
		? DENSE_ONLY : MERGE_ONLY;
	const double best_only = std::min(plan.merge_only_cycles, plan.dense_only_cycles);
	plan.predicted_adaptive_gain_cycles = std::max(0.0, best_only - plan.adaptive_cycles);
	if (plan.adaptive_cycles * model.global_adaptive_margin < best_only) {
		plan.mode = ADAPTIVE;
		plan.decision_reason = "adaptive-clears-global-margin";
	} else {
		plan.decision_reason = "best-only-or-insufficient-margin";
	}
	plan.model_mode = plan.mode;
	return plan;
}

// Final single-call guard.  Even a correct kernel-cycle prediction should not
// select mixed routing when Host analysis and route transfer cost more than the
// predicted gain.  Repetitions represent reuse of the same route table.
inline void account_for_host_overhead(Plan& plan, double selector_ms,
		unsigned repetitions, double route_transfer_ms = 0.03,
		double kernel_mhz = 200.0) {
	plan.selector_overhead_ms = selector_ms + route_transfer_ms;
	plan.predicted_gain_ms = plan.predicted_adaptive_gain_cycles
		/ (kernel_mhz * 1000.0);
	if (plan.mode == ADAPTIVE
			&& plan.predicted_gain_ms * std::max(1U, repetitions)
			<= plan.selector_overhead_ms) {
		plan.mode = plan.dense_only_cycles < plan.merge_only_cycles
			? DENSE_ONLY : MERGE_ONLY;
		plan.decision_reason = "adaptive-gain-below-host-overhead";
	}
}

} // namespace adaptive_host

#endif
