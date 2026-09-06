// U280 Host for the single-CU, eight-shard adaptive HBM SpGEMM kernel.
// Usage: app_adaptive_hbm_spgemm.exe xclbin device matrix.mtx results.csv [reps]

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "adaptive_light_selector.h"
#include "adaptive_route_model.h"
#include "adaptive_hbm/adaptive_hbm_route_word.h"
#include "host_device_utils.h"

namespace {

using Clock = std::chrono::steady_clock;
using id_t = uint32_t;

constexpr unsigned kShards = 8;
// Physical columns covered by one kernel invocation.  This is not a
// matrix-size or selector threshold: every matrix uses the same windowed
// execution path and receives ceil(N / kWindowColumns) column windows.
#ifdef ADAPT_RUNTIME_WINDOW_COLUMNS
// The dimension-command Host selects Tn before building the window index.  The
// legacy Host leaves ADAPT_RUNTIME_WINDOW_COLUMNS undefined and therefore
// retains the exact compile-time 64K behavior of the deployed checkpoint.
uint32_t kWindowColumns = uint32_t{1} << 16;
#else
constexpr uint32_t kWindowColumns = uint32_t{1} << 16;
#endif
constexpr unsigned kLocalMergeWays = 8;
constexpr unsigned kLocalMergeRunsPerShard = 2048;
constexpr unsigned kLocalMergePacketsPerBank = 256;
constexpr unsigned kLocalMergeRunItems = kLocalMergePacketsPerBank * 8;
// Entries [0, 12] remain the legacy ABI.  New TAPA bitstreams append the
// DENSE stage diagnostics and identify them with the DNS1 marker at index 13.
constexpr unsigned kStats = 43;
constexpr id_t kDenseDiagnosticVersion = 0x444e5331U;
constexpr unsigned kStatDiagnosticVersion = 13;
constexpr unsigned kStatDenseCollectCycles = 14;
constexpr unsigned kStatDenseEpochClearCycles = 15;
constexpr unsigned kStatDenseBankProducts = 16;
constexpr unsigned kStatDenseBankUniqueAddresses = 24;
constexpr unsigned kStatDenseTouchedAddresses = 32;
constexpr unsigned kStatDenseBitmapClearCycles = 33;
constexpr unsigned kStatDenseBitmapBuildCycles = 34;
constexpr unsigned kStatDenseBitmapWordScans = 35;
constexpr unsigned kStatDenseNonemptyWords = 36;
constexpr unsigned kStatDenseActiveAddressGroups = 37;
constexpr unsigned kStatDenseCandidateColumns = 38;
constexpr unsigned kStatDenseExtractIterations = 39;
constexpr unsigned kStatDenseOutputPackets = 40;
constexpr unsigned kStatMergeStreamCycles = 41;
constexpr unsigned kStatDenseCoalescedProducts = 42;
constexpr unsigned kRouteModeBits = ADAPT_HBM_ROUTE_MODE_BITS;
constexpr uint32_t kRouteModeMask = ADAPT_HBM_ROUTE_MODE_MASK;
constexpr uint32_t kRouteProductsMax = ADAPT_HBM_ROUTE_PRODUCTS_MAX;
constexpr uint32_t kScalableCapacityLock = 0x80000000U;
constexpr uint32_t kScalableProductsMax = 0x1fffffffU;
// A highly referenced super-long B row must not remain one indivisible HBM
// command.  Above this threshold, preserve sorted order inside 256-item runs
// and place those runs independently.  The kernel already treats every task as
// an ordered run, so both MERGE and DENSE consume this striping unchanged.
// The c-52 failure exposed a 1,714-item indivisible row; com-Youtube then
// exposed the same early-boundary condition in a different 64K window with a
// shorter but highly referenced row.  Make 256 the actual maximum physical B
// run, rather than a matrix-specific trigger, so no selector/window can
// recreate a single long command on one shard.
constexpr id_t kScalableBStripeThreshold = 256;
constexpr id_t kScalableBStripeItems = 256;

static uint64_t scalable_b_stripe_count(id_t length) {
	return length >= kScalableBStripeThreshold
		? (uint64_t(length) + kScalableBStripeItems - 1)
			/ kScalableBStripeItems
		: 1;
}

enum ForceMode : uint32_t { kUseRoute = 0, kForceMerge = 1, kForceDense = 2 };

struct Csr {
	id_t rows = 0;
	id_t cols = 0;
	std::vector<id_t> rowptr;
	std::vector<id_t> col;
	std::vector<float> val;
};

struct Reference {
	std::vector<id_t> rowptr;
	std::vector<id_t> col;
	std::vector<double> val;
	std::vector<double> abs_sum;
	std::vector<id_t> terms;
	uint64_t products = 0;
};

struct alignas(64) Beat512 { uint64_t lane[8]{}; };
struct alignas(64) TaskWord512 { uint64_t lane[8]{}; };

struct BLocation {
	id_t shard = 0;
	id_t beat_offset = 0;
	id_t length = 0;
};

struct PackedInput {
	std::vector<TaskWord512> tasks;
	std::vector<id_t> row_task_ptr;
	std::vector<id_t> route;
	std::array<std::vector<Beat512>, kShards> B;
	std::array<uint64_t, kShards> shard_work{};
	uint64_t max_local_merge_runs = 0;
	uint64_t max_local_merge_bank_packets = 0;
};

struct DmsaBatchSummary {
	uint64_t selected_rows = 0;
	uint64_t selected_products = 0;
	uint64_t fallback_rows = 0;
	uint64_t batches = 0;
};

struct ModeResult {
	std::string mode;
	double kernel_ms = 0.0;
	double d2h_ms = 0.0;
	double verification_ms = 0.0;
	std::vector<id_t> stats;
};

struct ScalableWindowSummary {
	uint32_t windows = 0;
	uint64_t total_window_tasks = 0;
	uint64_t total_products = 0;
	uint64_t output_nnz_upper_bound = 0;
	uint64_t max_window_tasks = 0;
	uint64_t max_window_products = 0;
	uint64_t max_window_output_upper_bound = 0;
	uint64_t max_b_beats_per_shard = 0;
	uint64_t max_row_window_runs = 0;
	uint64_t max_row_window_products = 0;
	uint64_t minimum_output_tiles = 0;
};

struct WindowSegment {
	uint32_t window = 0;
	id_t source_begin = 0;
	id_t length = 0;
};

struct WindowRowWork {
	id_t row = 0;
	uint64_t runs = 0;
	uint64_t products = 0;
};

struct ScalableIndex {
	ScalableWindowSummary summary;
	std::vector<std::vector<WindowSegment>> B_segments;
	std::vector<std::vector<WindowRowWork>> work_by_window;
};

struct WindowPackResult {
	PackedInput packed;
	adaptive_host::Plan plan;
	uint32_t window = 0;
	id_t row_begin = 0;
	id_t row_end = 0;
	uint64_t products = 0;
	uint64_t output_nnz_upper_bound = 0;
	uint64_t capacity_forced_dense_rows = 0;
	uint64_t conservative_capacity_locked_rows = 0;
};

struct ScalableExecutionResult {
	double packing_ms = 0.0;
	double h2d_ms = 0.0;
	double kernel_ms = 0.0;
	double d2h_ms = 0.0;
	double verification_ms = 0.0;
	uint64_t tiles = 0;
	uint64_t output_nnz = 0;
	uint64_t merge_rows = 0;
	uint64_t dense_rows = 0;
	uint64_t sampled_rows = 0;
	uint64_t default_merge_rows = 0;
	uint64_t default_dense_rows = 0;
	uint64_t default_merge_products = 0;
	uint64_t default_dense_products = 0;
	uint64_t capacity_forced_dense_rows = 0;
	uint64_t conservative_capacity_locked_rows = 0;
	uint64_t structural_crosscheck_rows = 0;
	uint64_t baseline_crosscheck_rows = 0;
};

struct RowStructureFingerprint {
	uint64_t first = 0;
	uint64_t second = 0;
};

static bool g_used_light_selector = false;
static adaptive_host::LightSelectorReport g_light_selector_report;

static std::string lower(std::string text) {
	std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
		return static_cast<char>(std::tolower(c));
	});
	return text;
}

static uint32_t float_bits(float value) {
	uint32_t raw = 0;
	std::memcpy(&raw, &value, sizeof(raw));
	return raw;
}

static float bits_float(uint32_t raw) {
	float value = 0.0f;
	std::memcpy(&value, &raw, sizeof(value));
	return value;
}

static Csr load_matrix_market(const std::string& path) {
	std::ifstream input(path);
	if (!input) throw std::runtime_error("cannot open matrix: " + path);
	std::string line;
	std::getline(input, line);
	const std::string header = lower(line);
	if (header.find("matrix coordinate") == std::string::npos
			|| header.find(" complex ") != std::string::npos)
		throw std::runtime_error("unsupported Matrix Market type");
	const bool pattern = header.find(" pattern ") != std::string::npos;
	const bool symmetric = header.find(" symmetric") != std::string::npos
		|| header.find(" hermitian") != std::string::npos;
	const bool skew = header.find(" skew-symmetric") != std::string::npos;
	do {
		if (!std::getline(input, line)) throw std::runtime_error("missing dimensions");
	} while (line.empty() || line[0] == '%');
	uint64_t rows = 0, cols = 0, entries = 0;
	std::stringstream dims(line);
	dims >> rows >> cols >> entries;
	if (!dims || rows == 0 || rows != cols
			|| rows > std::numeric_limits<id_t>::max())
		throw std::runtime_error("scalable kernel requires a square uint32 matrix");
	std::vector<std::vector<std::pair<id_t, float>>> per_row(rows);
	for (uint64_t index = 0; index < entries; ++index) {
		do {
			if (!std::getline(input, line)) throw std::runtime_error("truncated matrix");
		} while (line.empty() || line[0] == '%');
		uint64_t row = 0, col = 0;
		float value = 1.0f;
		std::stringstream item(line);
		item >> row >> col;
		if (!pattern) item >> value;
		if (!item || row == 0 || col == 0 || row > rows || col > cols)
			throw std::runtime_error("invalid Matrix Market entry");
		--row;
		--col;
		per_row[row].push_back({static_cast<id_t>(col), value});
		if ((symmetric || skew) && row != col)
			per_row[col].push_back({static_cast<id_t>(row), skew ? -value : value});
	}
	Csr matrix;
	matrix.rows = static_cast<id_t>(rows);
	matrix.cols = static_cast<id_t>(cols);
	matrix.rowptr.push_back(0);
	for (auto& row : per_row) {
		std::sort(row.begin(), row.end(), [](const auto& a, const auto& b) {
			return a.first < b.first;
		});
		for (size_t begin = 0; begin < row.size();) {
			size_t end = begin + 1;
			float value = row[begin].second;
			while (end < row.size() && row[end].first == row[begin].first)
				value += row[end++].second;
			if (value != 0.0f) {
				matrix.col.push_back(row[begin].first);
				matrix.val.push_back(value);
			}
			begin = end;
		}
		matrix.rowptr.push_back(static_cast<id_t>(matrix.col.size()));
	}
	return matrix;
}

static Reference square_reference(const Csr& matrix) {
	struct Accum {
		double value = 0.0;
		double abs_sum = 0.0;
		id_t terms = 0;
	};
	Reference ref;
	ref.rowptr.push_back(0);
	for (id_t row = 0; row < matrix.rows; ++row) {
		std::map<id_t, Accum> accum;
		for (id_t at = matrix.rowptr[row]; at < matrix.rowptr[row + 1]; ++at) {
			const id_t k = matrix.col[at];
			for (id_t bt = matrix.rowptr[k]; bt < matrix.rowptr[k + 1]; ++bt) {
				const double product = static_cast<double>(matrix.val[at])
					* static_cast<double>(matrix.val[bt]);
				Accum& entry = accum[matrix.col[bt]];
				entry.value += product;
				entry.abs_sum += std::fabs(product);
				entry.terms++;
				ref.products++;
			}
		}
		for (const auto& item : accum) {
			// Keep the complete structural product.  The FPGA is allowed to
			// prune a numerically-zero FP32 result; row-wise verification below
			// compares such a missing item against zero using a forward-error
			// bound instead of requiring identical CSR sparsity.
			ref.col.push_back(item.first);
			ref.val.push_back(item.second.value);
			ref.abs_sum.push_back(item.second.abs_sum);
			ref.terms.push_back(item.second.terms);
		}
		ref.rowptr.push_back(static_cast<id_t>(ref.col.size()));
	}
	return ref;
}

// Analyze the 2-D window/batch execution used by the scalable Host. Columns are
// converted to a 16-bit window-local key, preserving the timing-closed local
// MERGE comparator and making the same 64K DENSE accumulator reusable for
// arbitrary uint32 matrix dimensions.  Rows are batched only when a column
// window's conservative output bound exceeds four 256-MiB C banks.
static ScalableIndex build_scalable_index(const Csr& matrix) {
	constexpr uint64_t kHbmBankBytes = 256ULL << 20;
	constexpr uint64_t kOutputItemsPerTile
		= 4 * kHbmBankBytes / sizeof(uint64_t);
	const uint32_t windows = static_cast<uint32_t>(
		(matrix.cols + uint64_t(kWindowColumns) - 1) / kWindowColumns);

	ScalableIndex index;
	index.B_segments.resize(matrix.rows);
	index.work_by_window.resize(windows);
	std::vector<std::vector<id_t>> rows_by_window(windows);
	std::vector<uint64_t> references(matrix.rows, 0);
	for (id_t col : matrix.col)
		if (col < matrix.rows) references[col]++;
	for (id_t row = 0; row < matrix.rows; ++row) {
		const id_t begin = matrix.rowptr[row];
		const id_t end = matrix.rowptr[row + 1];
		for (id_t at = begin; at < end;) {
			const uint32_t window = matrix.col[at] / kWindowColumns;
			id_t next = at + 1;
			while (next < end
					&& matrix.col[next] / kWindowColumns == window)
				++next;
			index.B_segments[row].push_back(
				{window, at, static_cast<id_t>(next - at)});
			rows_by_window[window].push_back(row);
			at = next;
		}
	}

	ScalableWindowSummary& summary = index.summary;
	summary.windows = windows;
	std::vector<uint64_t> window_tasks(windows, 0);
	std::vector<uint64_t> window_products(windows, 0);
	std::vector<uint64_t> window_output_upper(windows, 0);
	for (id_t row = 0; row < matrix.rows; ++row) {
		std::map<uint32_t, std::array<uint64_t, 2>> work;
		for (id_t at = matrix.rowptr[row]; at < matrix.rowptr[row + 1]; ++at) {
			const id_t k = matrix.col[at];
			for (const WindowSegment& segment : index.B_segments[k]) {
				auto& value = work[segment.window];
				value[0] += scalable_b_stripe_count(segment.length);
				value[1] += segment.length;
			}
		}
		for (const auto& item : work) {
			const uint32_t window = item.first;
			const uint64_t runs = item.second[0];
			const uint64_t products = item.second[1];
			const uint32_t window_width = std::min<uint64_t>(kWindowColumns,
				uint64_t(matrix.cols) - uint64_t(window) * kWindowColumns);
			window_tasks[window] += runs;
			window_products[window] += products;
			window_output_upper[window]
				+= std::min<uint64_t>(products, window_width);
			index.work_by_window[window].push_back({row, runs, products});
			summary.max_row_window_runs = std::max(
				summary.max_row_window_runs, runs);
			summary.max_row_window_products = std::max(
				summary.max_row_window_products, products);
		}
	}

	for (uint32_t window = 0; window < windows; ++window) {
		std::vector<std::pair<uint64_t, uint32_t>> jobs;
		jobs.reserve(rows_by_window[window].size());
		for (id_t row : rows_by_window[window]) {
			const auto& segments = index.B_segments[row];
			const auto found = std::lower_bound(segments.begin(), segments.end(),
				window, [](const WindowSegment& segment, uint32_t key) {
					return segment.window < key;
				});
			if (found == segments.end() || found->window != window)
				throw std::runtime_error("internal window segment index mismatch");
			for (id_t offset = 0; offset < found->length;) {
				const id_t remaining = found->length - offset;
				const id_t length = found->length >= kScalableBStripeThreshold
					? std::min<id_t>(remaining, kScalableBStripeItems)
					: remaining;
				const uint32_t beats = (length + 7) / 8;
				const uint64_t weighted = std::max<uint64_t>(
					length, references[row] * length);
				jobs.push_back({weighted, beats});
				offset += length;
			}
		}
		std::sort(jobs.begin(), jobs.end(), [](const auto& left,
				const auto& right) { return left.first > right.first; });
		std::array<uint64_t, kShards> shard_work{};
		std::array<uint64_t, kShards> shard_beats{};
		for (const auto& job : jobs) {
			const unsigned shard = static_cast<unsigned>(std::min_element(
				shard_work.begin(), shard_work.end()) - shard_work.begin());
			shard_work[shard] += job.first;
			shard_beats[shard] += job.second;
		}
		summary.max_b_beats_per_shard = std::max(
			summary.max_b_beats_per_shard,
			*std::max_element(shard_beats.begin(), shard_beats.end()));
		summary.total_window_tasks += window_tasks[window];
		summary.total_products += window_products[window];
		summary.output_nnz_upper_bound += window_output_upper[window];
		summary.max_window_tasks = std::max(
			summary.max_window_tasks, window_tasks[window]);
		summary.max_window_products = std::max(
			summary.max_window_products, window_products[window]);
		summary.max_window_output_upper_bound = std::max(
			summary.max_window_output_upper_bound, window_output_upper[window]);
		summary.minimum_output_tiles += std::max<uint64_t>(1,
			(window_output_upper[window] + kOutputItemsPerTile - 1)
				/ kOutputItemsPerTile);
	}
	return index;
}

static bool choose_cheap_only(const Csr& matrix, adaptive_host::Plan* plan) {
	plan->route.resize(matrix.rows, 0);
	double product_sum = 0.0;
	double product_square_sum = 0.0;
	uint64_t multi_products = 0;
	uint32_t active_rows = 0;
	uint32_t multi_rows = 0;
	uint32_t pair_count = 0;
	uint32_t quad_count = 0;
	for (id_t row = 0; row < matrix.rows; ++row) {
		const id_t runs = matrix.rowptr[row + 1] - matrix.rowptr[row];
		uint64_t products = 0;
		id_t max_run = 0;
		for (id_t at = matrix.rowptr[row]; at < matrix.rowptr[row + 1]; ++at) {
			const id_t k = matrix.col[at];
			const id_t run = matrix.rowptr[k + 1] - matrix.rowptr[k];
			products += run;
			max_run = std::max(max_run, run);
		}
		if (runs == 0 || products == 0) {
			plan->route[row] = 0;
			++plan->empty_rows;
			continue;
		}
		product_sum += products;
		product_square_sum += static_cast<double>(products) * products;
		++active_rows;
		if (runs == 1) {
			plan->route[row] = 1;
			++plan->direct_rows;
		} else {
			plan->route[row] = 2;
			++plan->merge_rows;
			plan->merge_products += products;
			++multi_rows;
			multi_products += products;
			++plan->multi_rows;
			plan->multi_products += products;
			if (runs <= adaptive_host::kMergeWidth
					&& max_run <= adaptive_host::kWideMergeMaxRun) {
				++plan->wide_rows;
				plan->wide_products += products;
			}
		}
	}
	for (id_t row = 0; row < matrix.rows;) {
		const bool quad = matrix.cols <= adaptive_host::kDenseQuadCapacity
			&& (row & 3U) == 0 && row + 3 < matrix.rows
			&& plan->route[row] == 2 && plan->route[row + 1] == 2
			&& plan->route[row + 2] == 2 && plan->route[row + 3] == 2;
		if (quad) {
			++quad_count;
			row += 4;
			continue;
		}
		const bool pair = (row & 1U) == 0 && row + 1 < matrix.rows
			&& plan->route[row] == 2 && plan->route[row + 1] == 2;
		if (pair) {
			++pair_count;
			row += 2;
			continue;
		}
		++row;
	}
	plan->dense_only_pairs = pair_count;
	plan->dense_only_quads = quad_count;
	if (active_rows != 0 && product_sum != 0.0) {
		const double mean = product_sum / active_rows;
		plan->active_product_cv = std::sqrt(std::max(0.0,
			product_square_sum / active_rows - mean * mean)) / mean;
	}
	if (multi_rows == 0) {
		plan->mode = plan->model_mode = adaptive_host::MERGE_ONLY;
		plan->decision_reason = "cheap-no-reduction-work";
		return true;
	}
	if (plan->empty_rows == 0 && plan->direct_rows == 0
			&& plan->active_product_cv <= 0.50) {
		plan->mode = plan->model_mode = adaptive_host::DENSE_ONLY;
		plan->decision_reason = "cheap-regular-low-variance-dense-only";
		return true;
	}
	const double wide_fraction = static_cast<double>(plan->wide_products)
		/ multi_products;
	const double quad_coverage = static_cast<double>(quad_count) * 4.0
		/ multi_rows;
	const double products_per_multi_row =
		static_cast<double>(multi_products) / multi_rows;

	// Below this work level, detailed HLL selection costs more than the entire
	// path difference (0.4--1.0 ms selector time in the holdout runs).  If all
	// work is wide-eligible and no aligned quad exists, routed merge has the
	// lower measured launch-to-completion time; otherwise dense remains the
	// low-overhead small-work default.
	if (multi_products <= 4096U) {
		const bool use_merge = wide_fraction >= 0.90 && quad_count == 0;
		plan->mode = plan->model_mode = use_merge
			? adaptive_host::MERGE_ONLY : adaptive_host::DENSE_ONLY;
		plan->decision_reason = use_merge
			? "cheap-small-wide-no-quad-merge-only"
			: "cheap-small-work-dense-only";
		return true;
	}

	// If at most one quarter of product work can enter the 16-way wide path,
	// the legacy merge fallback dominates.  The quad dense reducer is the
	// board-tested winner in this region.
	if (wide_fraction <= 0.25) {
		plan->mode = plan->model_mode = adaptive_host::DENSE_ONLY;
		plan->decision_reason = "cheap-wide-ineligible-work-dense-only";
		return true;
	}

	// An aligned four-row batch shares one dense invocation, tag/bitmap setup,
	// and four independent 8-bank contexts.  Near-complete quad coverage is a
	// strong dense-only signal even when most rows also satisfy wide merge.
	if (quad_coverage >= 0.875) {
		plan->mode = plan->model_mode = adaptive_host::DENSE_ONLY;
		plan->decision_reason = "cheap-quad-dominant-dense-only";
		return true;
	}

	// For short multi-run rows, even partial quad coverage amortizes dense
	// setup more effectively than starting a merge tree per row.
	if (quad_coverage >= 0.25 && products_per_multi_row <= 128.0) {
		plan->mode = plan->model_mode = adaptive_host::DENSE_ONLY;
		plan->decision_reason = "cheap-short-quad-dense-only";
		return true;
	}

	// A nearly all-wide matrix does not need a large-work guard: the independent
	// loaders and DIRECT/EMPTY bypass already dominate the forced dense path.
	// In the balanced-wide region, require at least 2^20 products before taking
	// the same cheap exit.  Below that crossover, measured submatrices still
	// favor dense, so the detailed model must inspect replay/materialization.
	if (wide_fraction >= 0.90
			|| (wide_fraction >= 0.35 && multi_products >= (1ULL << 20))) {
		plan->mode = plan->model_mode = adaptive_host::MERGE_ONLY;
		plan->decision_reason = "cheap-wide-work-merge-only";
		return true;
	}

	// The remaining transition region is intentionally left to the HLL32,
	// bank-replay, and quad-aware detailed model.
	return false;
}

static adaptive_host::Plan choose_routes(const Csr& matrix) {
	// Lightweight online selector derived from the conditional analysis in
	// spECK and sampled compression-ratio estimation in Ocean.  This is the
	// production default for the TAPA checkpoint: the mandatory pass is O(nnz)
	// and only ambiguous feature bins receive a bounded HLL sample.  Retain an
	// explicit escape hatch for reproducing the old detailed cycle predictor.
	const char* light_selector = std::getenv("ADAPT_GPU_LIGHT_SELECTOR");
	const bool force_detailed =
		std::getenv("ADAPT_USE_DETAILED_SELECTOR") != nullptr;
	const bool use_light_selector = !force_detailed
		&& (light_selector == nullptr || std::strcmp(light_selector, "0") != 0);
	if (use_light_selector) {
		adaptive_host::LightSelectorConfig config;
		if (const char* value = std::getenv("ADAPT_LIGHT_DENSE_CR")) {
			const double parsed = std::strtod(value, nullptr);
			if (std::isfinite(parsed) && parsed >= 1.0 && parsed <= 128.0)
				config.dense_min_compression = parsed;
		}
		if (const char* value = std::getenv("ADAPT_LIGHT_MIN_PRODUCTS")) {
			const unsigned long long parsed = std::strtoull(value, nullptr, 10);
			if (parsed >= 64 && parsed <= kRouteProductsMax)
				config.dense_min_products = parsed;
		}
		if (const char* value = std::getenv("ADAPT_LIGHT_SAMPLE_RATE")) {
			const double parsed = std::strtod(value, nullptr);
			if (std::isfinite(parsed) && parsed >= 0.0 && parsed <= 0.20)
				config.sample_rate = parsed;
		}
		if (std::getenv("ADAPT_LIGHT_DISABLE_SAMPLING") != nullptr)
			config.enable_sampling = false;
		adaptive_host::Plan plan = adaptive_host::choose_light_plan(
			matrix.rowptr, matrix.col, matrix.cols, config,
			&g_light_selector_report);
		g_used_light_selector = true;

		// Board calibration for the 160-MHz DENSE B-row-cache checkpoint.  Its
		// real3 oracle campaign separates two genuinely different regimes:
		// s3rmt3m3/bcsstk24 route 81--86% of product work to DENSE but still favor
		// MERGE_ONLY, whereas cavity26 routes only 14.6% of the work (376 high-work
		// rows) to DENSE and ROW_ADAPTIVE is 2.13% faster than MERGE_ONLY.  Admit
		// only that structurally narrow mixed regime by default: MERGE rows must
		// dominate by at least 4:1 while DENSE still carries 10--20% of product
		// work and at least 128 rows.  This is an O(1) decision over counters that
		// choose_light_plan() already produced; it adds no second prediction pass.
		const bool calibrated_sparse_dense_mix =
			plan.mode == adaptive_host::ADAPTIVE
			&& g_light_selector_report.dense_work_fraction >= 0.10
			&& g_light_selector_report.dense_work_fraction <= 0.20
			&& plan.dense_rows >= 128
			&& static_cast<uint64_t>(plan.merge_rows)
				>= 4ULL * plan.dense_rows;
		const bool calibrated_exact_bimodal_mix =
			plan.mode == adaptive_host::ADAPTIVE
			&& g_light_selector_report.ambiguous_rows == 0
			&& g_light_selector_report.hll_build_entries == 0
			&& g_light_selector_report.dense_work_fraction >= 0.75
			&& g_light_selector_report.dense_work_fraction <= 0.90
			&& plan.dense_rows >= 1024
			&& static_cast<uint64_t>(plan.merge_rows)
				>= 2ULL * plan.dense_rows;
		const bool calibrated_adaptive_mix = calibrated_sparse_dense_mix
			|| calibrated_exact_bimodal_mix;
		const char* allow_uncalibrated =
			std::getenv("ADAPT_ENABLE_UNCALIBRATED_DENSE");
		const bool calibrated_merge_gate = allow_uncalibrated == nullptr
			|| std::strcmp(allow_uncalibrated, "0") == 0;
		if ((calibrated_merge_gate && !calibrated_adaptive_mix)
				|| std::getenv("ADAPT_TAPA_PARALLEL_SELECTOR") != nullptr) {
			plan.mode = adaptive_host::MERGE_ONLY;
			plan.model_mode = adaptive_host::MERGE_ONLY;
			plan.predicted_adaptive_gain_cycles = 0.0;
			plan.decision_reason = "dense-bcache-real3-merge-safety-gate";
		} else if (calibrated_merge_gate && calibrated_sparse_dense_mix) {
			plan.decision_reason = "dense-bcache-sparse-dense-mix-calibration";
		} else if (calibrated_merge_gate && calibrated_exact_bimodal_mix) {
			plan.decision_reason = "dense-bcache-exact-bimodal-mix-calibration";
		}
		return plan;
	}

	// The old cheap exits were calibrated for the pre-whole-cache RTL.  They
	// collapse every paper-10 matrix to DENSE_ONLY before the detailed model is
	// even evaluated, so they are opt-in for old-bitstream reproduction only.
	// The current profile always computes real per-row MERGE/DENSE features.
	const char* legacy_cheap = std::getenv("ADAPT_USE_LEGACY_CHEAP_SELECTOR");
	if (legacy_cheap != nullptr && std::strcmp(legacy_cheap, "0") != 0
			&& std::getenv("ADAPT_BENCHMARK_ROUTES") == nullptr) {
		adaptive_host::Plan cheap;
		if (choose_cheap_only(matrix, &cheap)) return cheap;
	}
	uint64_t work = 0;
	const auto features = adaptive_host::extract_matrix_features(
		matrix.rowptr, matrix.col, matrix.cols, &work);
	adaptive_host::CostModel model;
	// Total-cycle predictors calibrated from real RTL rows.  The important
	// distinction is that merge materializes P products serially, whereas the
	// grouped HBM path produces them through eight independent shard loaders.
	// Treating P as common work was the stale v1 model's main false-negative
	// source on poisson3Da, bitcoin-alpha and wiki-Vote.
	model.producer_per_product = 0.0;
	model.output_per_nnz = 0.0;
	model.merge_per_product = 3.0462016513;
	model.merge_max_run = 26.5310879406;
	model.merge_critical = 0.0;
	model.merge_materialized = 10.7640965427;
	model.merge_final_copy = 0.0;
	model.merge_transaction = 198.423883492;
	model.dense_per_product = 0.5624710195;
	model.dense_round_ii = 0.0;
	model.dense_bank_replay = 8.7627963542;
	model.dense_extract_per_nnz = 3.3399254595;
	model.dense_row_setup = 1376.45516912;
	model.dense_pair_setup_credit = model.dense_row_setup;
	model.dense_quad_setup_credit = 3.0 * model.dense_row_setup;
	model.dense_oct_setup_credit = 7.0 * model.dense_row_setup;
	model.dense_hex_setup_credit = 15.0 * model.dense_row_setup;
	model.row_dense_margin = 1.07;
	model.dense_min_runs = 17;
	model.dense_short_row_penalty = 1.08;
	if (const char* margin = std::getenv("ADAPT_DENSE_MARGIN")) {
		const double value = std::strtod(margin, nullptr);
		if (std::isfinite(value) && value >= 1.0 && value <= 2.0)
			model.row_dense_margin = value;
	}
	adaptive_host::Plan plan = adaptive_host::choose_plan(
		features, matrix.cols, model, kWindowColumns, work);

	// Board-calibrated correction for the current whole-cache row-RR datapath.
	// Its additive row model overcharges OCT-mode batch fragmentation when a
	// high-variance matrix has a modest set of cheap MERGE rows.  On the formal
	// paper-10 run this is exactly the poisson3Da false negative: the predicted
	// mixed/dense ratio is 1.140, while the measured ratio is 0.895.  Keep the
	// correction deliberately narrow so t2dah_a (many MERGE rows), af23560
	// (low variance), and HEX-mode matrices do not inherit that decision.
	const uint32_t routed_rows = plan.merge_rows + plan.dense_rows;
	const double merge_row_fraction = routed_rows == 0 ? 0.0
		: static_cast<double>(plan.merge_rows) / routed_rows;
	const bool calibrated_oct_mix = matrix.cols > 8192
		&& matrix.cols <= adaptive_host::kDenseOctCapacity
		&& plan.active_product_cv >= 0.50
		&& merge_row_fraction >= 0.05 && merge_row_fraction <= 0.20
		&& plan.adaptive_cycles <= 1.15 * plan.dense_only_cycles;
	if (calibrated_oct_mix
			&& std::getenv("ADAPT_DISABLE_ROWRR_OCT_CALIBRATION") == nullptr) {
		// The 0.90 factor is conservative relative to the measured 0.895 ratio.
		// It also supplies account_for_host_overhead() with a real predicted gain,
		// so a one-shot call may still reject ADAPTIVE when selector cost dominates.
		plan.adaptive_cycles = 0.90 * plan.dense_only_cycles;
		plan.predicted_adaptive_gain_cycles = 0.10 * plan.dense_only_cycles;
		plan.mode = adaptive_host::ADAPTIVE;
		plan.model_mode = adaptive_host::ADAPTIVE;
		plan.decision_reason = "rowrr-oct-high-cv-calibration";
	}

	// The TAPA parallel checkpoint has a different cycle balance from the
	// earlier whole-cache RTL used to fit the model above.  Its first complete
	// U280 paper-10 campaign showed MERGE_ONLY as the oracle on all ten
	// matrices, while the stale model selected DENSE/ADAPTIVE and incurred up
	// to 8.8x regret.  Keep this board-calibrated safety gate opt-in so older
	// bitstreams retain their original selector.  Row tags are left intact for
	// the explicit row_adaptive diagnostic mode; the host-selected invocation
	// uses the force-merge control and therefore still preserves the one unified
	// kernel rather than selecting another backend.
	if (std::getenv("ADAPT_TAPA_PARALLEL_SELECTOR") != nullptr) {
		plan.mode = adaptive_host::MERGE_ONLY;
		plan.model_mode = adaptive_host::MERGE_ONLY;
		plan.predicted_adaptive_gain_cycles = 0.0;
		plan.decision_reason = "tapa-parallel-r10-merge-safety-gate";
	}

	// Keep the mixed route available for measurement even when an only-mode is
	// predicted globally, but never force ADAPTIVE merely because both row
	// classes occur.  Dense batching is a non-additive benefit: isolated MERGE
	// rows can break 16/8/4-row DENSE batches, so only the batch-aware total-cost
	// comparison in choose_plan() is allowed to select the mixed mode.
	return plan;
}

static PackedInput pack_input(const Csr& matrix,
		const adaptive_host::Plan& plan) {
	PackedInput packed;
	packed.route.assign(plan.route.begin(), plan.route.end());
	packed.row_task_ptr.resize(matrix.rows + 1, 0);
	std::vector<uint64_t> references(matrix.rows, 0);
	for (id_t col : matrix.col) if (col < matrix.rows) references[col]++;
	std::vector<id_t> order(matrix.rows);
	std::iota(order.begin(), order.end(), 0);
	std::sort(order.begin(), order.end(), [&](id_t left, id_t right) {
		const uint64_t left_work = references[left]
			* (matrix.rowptr[left + 1] - matrix.rowptr[left]);
		const uint64_t right_work = references[right]
			* (matrix.rowptr[right + 1] - matrix.rowptr[right]);
		return left_work > right_work;
	});
	std::vector<BLocation> location(matrix.rows);
	for (id_t row : order) {
		const id_t length = matrix.rowptr[row + 1] - matrix.rowptr[row];
		const uint64_t work = std::max<uint64_t>(length, references[row] * length);
		const unsigned shard = static_cast<unsigned>(std::min_element(
			packed.shard_work.begin(), packed.shard_work.end())
			- packed.shard_work.begin());
		location[row] = {static_cast<id_t>(shard),
			static_cast<id_t>(packed.B[shard].size()), length};
		packed.shard_work[shard] += work;
		const id_t beats = (length + 7) / 8;
		for (id_t beat = 0; beat < beats; ++beat) {
			Beat512 word{};
			for (unsigned lane = 0; lane < 8; ++lane) {
				const id_t local = beat * 8 + lane;
				if (local < length) {
					const id_t index = matrix.rowptr[row] + local;
					word.lane[lane] = (uint64_t(float_bits(matrix.val[index])) << 32)
						| matrix.col[index];
				}
			}
			packed.B[shard].push_back(word);
		}
	}
	for (auto& shard : packed.B) if (shard.empty()) shard.resize(1);

	std::vector<std::array<uint64_t, 2>> descriptors;
	for (id_t row = 0; row < matrix.rows; ++row) {
		packed.row_task_ptr[row] = static_cast<id_t>(descriptors.size());
		const id_t begin = matrix.rowptr[row];
		const id_t end = matrix.rowptr[row + 1];
		uint64_t row_products = 0;
		std::array<std::vector<id_t>, kShards> local_merge_runs;
		for (id_t at = begin; at < end; ++at) {
			const id_t k = matrix.col[at];
			const BLocation loc = location[k];
			row_products += loc.length;
			// The reader closes a MERGE run every 256 packets.  Mirror that
			// long-run striping here so the Host checks the physical local-bank
			// layout, not the obsolete whole-row runs/products thresholds.
			id_t remaining = loc.length;
			while (remaining != 0) {
				const id_t chunk = std::min<id_t>(remaining,
					kLocalMergeRunItems);
				local_merge_runs[loc.shard].push_back(chunk);
				remaining -= chunk;
			}
			const bool last = at + 1 == end;
			uint64_t low = uint64_t(loc.beat_offset)
				| (uint64_t(loc.length) << 32)
				| (uint64_t(loc.shard) << 56)
				| (uint64_t(plan.route[row] & 3U) << 59)
				| (uint64_t(row & 3U) << 61)
				| (uint64_t(last) << 63);
			uint64_t high = uint64_t(row)
				| (uint64_t(float_bits(matrix.val[at])) << 32);
			descriptors.push_back({low, high});
		}
		packed.row_task_ptr[row + 1] = static_cast<id_t>(descriptors.size());
		// Every benchmark invocation also measures forced MERGE.  Validate all
		// multi-task rows against the exact striped run placement and the same
		// greedy, at-most-eight-way hierarchy used in the kernel.
		if (end - begin > 1) {
			for (unsigned shard = 0; shard < kShards; ++shard) {
				std::vector<id_t> current = local_merge_runs[shard];
				packed.max_local_merge_runs = std::max<uint64_t>(
					packed.max_local_merge_runs, current.size());
				if (current.size() > kLocalMergeRunsPerShard)
					throw std::runtime_error("row " + std::to_string(row)
						+ " shard " + std::to_string(shard) + " requires "
						+ std::to_string(current.size())
						+ " striped MERGE runs, exceeding "
						+ std::to_string(kLocalMergeRunsPerShard));
				unsigned stage = 0;
				while (!current.empty()) {
					std::array<uint64_t, kLocalMergeWays> bank_packets{};
					for (size_t run = 0; run < current.size(); ++run)
						bank_packets[run & (kLocalMergeWays - 1)]
							+= (current[run] + 7) / 8;
					const uint64_t stage_max = *std::max_element(
						bank_packets.begin(), bank_packets.end());
					packed.max_local_merge_bank_packets = std::max(
						packed.max_local_merge_bank_packets, stage_max);
					if (stage_max > kLocalMergePacketsPerBank)
						throw std::runtime_error("row " + std::to_string(row)
							+ " shard " + std::to_string(shard) + " MERGE stage "
							+ std::to_string(stage) + " requires "
							+ std::to_string(stage_max)
							+ " packets in one bank, exceeding "
							+ std::to_string(kLocalMergePacketsPerBank));
					if (current.size() <= kLocalMergeWays) break;
					std::vector<id_t> next;
					for (size_t run = 0; run < current.size();) {
						id_t group_items = 0;
						unsigned group_runs = 0;
						while (run < current.size()
								&& group_runs < kLocalMergeWays
								&& group_items + current[run]
									<= kLocalMergeRunItems) {
							group_items += current[run++];
							++group_runs;
						}
						if (group_runs == 0)
							throw std::runtime_error("internal MERGE capacity model stalled");
						next.push_back(group_items);
					}
					current.swap(next);
					++stage;
				}
			}
		}
		// Low two bits remain wire-compatible with every deployed XCLBIN.
		// Upper bits prepare exact, oracle-free work metadata for the selected
		// DMSA batch gate; current kernels simply ignore them.
		const uint32_t bounded_products = static_cast<uint32_t>(
			std::min<uint64_t>(row_products, kRouteProductsMax));
		packed.route[row] = (plan.route[row] & kRouteModeMask)
			| (bounded_products << kRouteModeBits);
	}
	packed.tasks.resize(std::max<size_t>(1, (descriptors.size() + 3) / 4));
	for (size_t index = 0; index < descriptors.size(); ++index) {
		packed.tasks[index / 4].lane[(index & 3) * 2] = descriptors[index][0];
		packed.tasks[index / 4].lane[(index & 3) * 2 + 1] = descriptors[index][1];
	}
	return packed;
}

static bool local_merge_capacity_fits(std::vector<id_t> lengths,
		uint64_t* max_runs, uint64_t* max_bank_packets) {
	std::vector<id_t> striped;
	for (id_t length : lengths)
		while (length != 0) {
			const id_t chunk = std::min<id_t>(length, kLocalMergeRunItems);
			striped.push_back(chunk);
			length -= chunk;
		}
	*max_runs = std::max<uint64_t>(*max_runs, striped.size());
	if (striped.size() > kLocalMergeRunsPerShard) return false;
	while (!striped.empty()) {
		std::array<uint64_t, kLocalMergeWays> bank_packets{};
		for (size_t run = 0; run < striped.size(); ++run)
			bank_packets[run & (kLocalMergeWays - 1)]
				+= (striped[run] + 7) / 8;
		const uint64_t stage_max = *std::max_element(
			bank_packets.begin(), bank_packets.end());
		*max_bank_packets = std::max(*max_bank_packets, stage_max);
		if (stage_max > kLocalMergePacketsPerBank) return false;
		if (striped.size() <= kLocalMergeWays) break;
		std::vector<id_t> next;
		for (size_t run = 0; run < striped.size();) {
			id_t group_items = 0;
			unsigned group_runs = 0;
			while (run < striped.size() && group_runs < kLocalMergeWays
					&& group_items + striped[run] <= kLocalMergeRunItems) {
				group_items += striped[run++];
				++group_runs;
			}
			if (group_runs == 0) return false;
			next.push_back(group_items);
		}
		if (next.size() >= striped.size()) return false;
		striped.swap(next);
	}
	return true;
}

// Pack one (row batch, 64K output-column window) execution region. B rows are sliced
// before sharding, and every stored column is rebased to a 16-bit window-local
// key.  This preserves the timing-closed MERGE/DENSE hardware while making
// matrix dimension independent of the accumulator width.
static WindowPackResult pack_window_tile(const Csr& matrix,
		const ScalableIndex& index, uint32_t window,
		id_t row_begin, id_t row_end) {
	const uint64_t column_base = uint64_t(window) * kWindowColumns;
	const uint64_t column_end = std::min<uint64_t>(
		matrix.cols, column_base + kWindowColumns);
	const bool debug_progress = std::getenv("ADAPT_WINDOW_PROGRESS") != nullptr;
	if (column_base >= column_end || row_begin >= row_end
			|| row_end > matrix.rows)
		throw std::runtime_error("invalid scalable window/batch region");

	struct SegmentLocation {
		struct Stripe {
			id_t source_begin = 0;
			BLocation location;
		};
		id_t source_begin = 0;
		id_t length = 0;
		std::vector<Stripe> stripes;
	};
	std::vector<SegmentLocation> segments(matrix.rows);
	std::vector<uint64_t> references(matrix.rows, 0);
	for (id_t row = row_begin; row < row_end; ++row)
		for (id_t at = matrix.rowptr[row]; at < matrix.rowptr[row + 1]; ++at)
			if (matrix.col[at] < matrix.rows) references[matrix.col[at]]++;
	std::vector<id_t> order;
	for (id_t row = 0; row < matrix.rows; ++row) {
		if (references[row] == 0) continue;
		const auto& row_segments = index.B_segments[row];
		const auto found = std::lower_bound(row_segments.begin(),
			row_segments.end(), window,
			[](const WindowSegment& segment, uint32_t key) {
				return segment.window < key;
			});
		if (found == row_segments.end() || found->window != window) continue;
		segments[row].source_begin = found->source_begin;
		segments[row].length = found->length;
		order.push_back(row);
	}
	std::sort(order.begin(), order.end(), [&](id_t left, id_t right) {
		return references[left] * segments[left].length
			> references[right] * segments[right].length;
	});

	WindowPackResult result;
	result.window = window;
	result.row_begin = row_begin;
	result.row_end = row_end;
	PackedInput& packed = result.packed;
	for (id_t row : order) {
		for (id_t stripe_offset = 0; stripe_offset < segments[row].length;) {
			const id_t remaining = segments[row].length - stripe_offset;
			const id_t length
				= segments[row].length >= kScalableBStripeThreshold
				? std::min<id_t>(remaining, kScalableBStripeItems)
				: remaining;
			const uint64_t work = std::max<uint64_t>(
				length, references[row] * length);
			const unsigned shard = static_cast<unsigned>(std::min_element(
				packed.shard_work.begin(), packed.shard_work.end())
				- packed.shard_work.begin());
			const id_t source_begin
				= segments[row].source_begin + stripe_offset;
			segments[row].stripes.push_back({source_begin,
				{static_cast<id_t>(shard),
				 static_cast<id_t>(packed.B[shard].size()), length}});
			packed.shard_work[shard] += work;
			for (id_t beat = 0; beat < (length + 7) / 8; ++beat) {
				Beat512 word{};
				for (unsigned lane = 0; lane < 8; ++lane) {
					const id_t local = beat * 8 + lane;
					if (local < length) {
						const id_t index = source_begin + local;
						word.lane[lane]
							= (uint64_t(float_bits(matrix.val[index])) << 32)
							| (matrix.col[index] - column_base);
					}
				}
				packed.B[shard].push_back(word);
			}
			stripe_offset += length;
		}
	}
	for (auto& shard : packed.B) if (shard.empty()) shard.resize(1);

	const id_t tile_rows = row_end - row_begin;
	packed.row_task_ptr.resize(tile_rows + 1, 0);
	packed.route.resize(tile_rows, 0);
	result.plan.route.resize(tile_rows, 0);
	std::vector<std::array<uint64_t, 2>> descriptors;
	for (id_t local_row = 0; local_row < tile_rows; ++local_row) {
		if (debug_progress && (local_row & 8191U) == 0)
			std::cerr << "window_pack task_rows window=" << window
				<< " row=" << local_row << '/' << tile_rows << '\n';
		const id_t row = row_begin + local_row;
		packed.row_task_ptr[local_row] = static_cast<id_t>(descriptors.size());
		std::array<std::vector<id_t>, kShards> shard_lengths;
		uint64_t products = 0;
		id_t min_col = kWindowColumns;
		id_t max_col = 0;
		for (id_t at = matrix.rowptr[row]; at < matrix.rowptr[row + 1]; ++at) {
			const id_t k = matrix.col[at];
			const SegmentLocation& segment = segments[k];
			if (segment.length == 0 || references[k] == 0) continue;
			min_col = std::min<id_t>(min_col,
				matrix.col[segment.source_begin] - column_base);
			max_col = std::max<id_t>(max_col,
				matrix.col[segment.source_begin + segment.length - 1] - column_base);
			for (const SegmentLocation::Stripe& stripe : segment.stripes) {
				const BLocation loc = stripe.location;
				products += loc.length;
				shard_lengths[loc.shard].push_back(loc.length);
				const bool last_placeholder = false;
				uint64_t low = uint64_t(loc.beat_offset)
					| (uint64_t(loc.length) << 32)
					| (uint64_t(loc.shard) << 56)
					| (uint64_t(local_row & 3U) << 61)
					| (uint64_t(last_placeholder) << 63);
				const uint64_t high = uint64_t(local_row)
					| (uint64_t(float_bits(matrix.val[at])) << 32);
				descriptors.push_back({low, high});
			}
		}
		const id_t task_begin = packed.row_task_ptr[local_row];
		const id_t task_end = static_cast<id_t>(descriptors.size());
		const id_t runs = task_end - task_begin;
		if (debug_progress && local_row < 128)
			std::cerr << "window_pack row_capacity_begin window=" << window
				<< " row=" << local_row << " runs=" << runs
				<< " products=" << products << '\n';
		uint32_t route = runs == 0 ? 0 : (runs == 1 ? 1 : 2);
		bool capacity_locked = false;
		if (runs > 1) {
			const uint64_t span = uint64_t(max_col) - min_col + 1;
			const bool dense_by_reuse = runs >= 17 && products >= 512
				&& static_cast<double>(products) / std::max<uint64_t>(1, span)
					>= 8.0;
			// Width remains a performance hint for the default adaptive route, not
			// a safety lock.  The physical striped hierarchy is cheap enough to
			// simulate even for c-52's 1,714-run row, so only an exact capacity or
			// progress failure may prevent the forced-MERGE oracle comparison.
			const bool dense_by_width = runs > 512 || products > 32768;
			bool merge_fits = true;
			for (unsigned shard = 0; shard < kShards; ++shard)
				merge_fits = local_merge_capacity_fits(shard_lengths[shard],
					&packed.max_local_merge_runs,
					&packed.max_local_merge_bank_packets) && merge_fits;
			if (!merge_fits || dense_by_reuse || dense_by_width) route = 3;
			if (!merge_fits) {
				result.capacity_forced_dense_rows++;
				capacity_locked = true;
			}
		}
		if (debug_progress && local_row < 128)
			std::cerr << "window_pack row_capacity_end window=" << window
				<< " row=" << local_row << " route=" << route << '\n';
		for (id_t task = task_begin; task < task_end; ++task) {
			descriptors[task][0] |= uint64_t(route) << 59;
			descriptors[task][0] |= uint64_t(task + 1 == task_end) << 63;
		}
		const uint32_t bounded_products = static_cast<uint32_t>(
			std::min<uint64_t>(products, kScalableProductsMax));
		packed.route[local_row] = route
			| (bounded_products << kRouteModeBits)
			| (capacity_locked ? kScalableCapacityLock : 0U);
		result.plan.route[local_row] = route;
		result.products += products;
		result.output_nnz_upper_bound += std::min<uint64_t>(
			products, column_end - column_base);
		if (route == 0) result.plan.empty_rows++;
		else if (route == 1) result.plan.direct_rows++;
		else if (route == 2) {
			result.plan.merge_rows++;
			result.plan.merge_products += products;
		} else {
			result.plan.dense_rows++;
			result.plan.dense_products += products;
		}
	}
	packed.row_task_ptr[tile_rows] = static_cast<id_t>(descriptors.size());
	packed.tasks.resize(std::max<size_t>(1, (descriptors.size() + 3) / 4));
	for (size_t index = 0; index < descriptors.size(); ++index) {
		packed.tasks[index / 4].lane[(index & 3) * 2] = descriptors[index][0];
		packed.tasks[index / 4].lane[(index & 3) * 2 + 1] = descriptors[index][1];
	}
	result.plan.mode = result.plan.dense_rows == 0
		? adaptive_host::MERGE_ONLY
		: (result.plan.merge_rows == 0 ? adaptive_host::DENSE_ONLY
			: adaptive_host::ADAPTIVE);
	result.plan.model_mode = result.plan.mode;
	return result;
}

static std::vector<std::pair<id_t, id_t>> make_window_row_batches(
		const Csr& matrix, const ScalableIndex& index, uint32_t window) {
	// Four physical 256-MiB output banks hold 134,217,728 FP32 (col,value)
	// items.  The packed task descriptors have an independent 256-MiB HBM bank:
	// four 128-bit descriptors occupy one 512-bit word, hence its exact ceiling
	// is 16,777,216 descriptors.  Limit both resources here.  com-LiveJournal
	// exposed why output-only batching is insufficient: a 120M-output region in
	// window 11 contained 22.5M short task descriptors and failed its task BO
	// allocation even though every C bank remained within capacity.
	//
	// Keep conservative headroom so an odd boundary can be extended by one row
	// to preserve DENSE pair alignment without risking either physical limit.
	constexpr uint64_t kTargetItems = 120000000;
	constexpr uint64_t kTargetTasks = 15000000;
	const uint64_t column_base = uint64_t(window) * kWindowColumns;
	const uint64_t column_end = std::min<uint64_t>(
		matrix.cols, column_base + kWindowColumns);
	std::vector<std::pair<id_t, id_t>> batches;
	id_t begin = 0;
	uint64_t batch_items = 0;
	uint64_t batch_tasks = 0;
	for (const WindowRowWork& work : index.work_by_window.at(window)) {
		const id_t row = work.row;
		const uint64_t upper = std::min<uint64_t>(
			work.products, column_end - column_base);
		if (row != begin && (batch_items + upper > kTargetItems
				|| batch_tasks + work.runs > kTargetTasks)) {
			id_t end = row;
			// Both begin and end stay even, retaining the reducer's physical
			// (0,1),(2,3),... DENSE pairing inside every tile.
			if (end & 1U) {
				batch_items += upper;
				batch_tasks += work.runs;
				end++;
			}
			batches.push_back({begin, std::min(end, matrix.rows)});
			begin = std::min(end, matrix.rows);
			// At an even split the current row begins the new batch.  At an
			// odd split it was deliberately included in the previous batch.
			batch_items = end == row ? upper : 0;
			batch_tasks = end == row ? work.runs : 0;
			if (begin >= matrix.rows) break;
			continue;
		}
		batch_items += upper;
		batch_tasks += work.runs;
	}
	if (begin < matrix.rows) batches.push_back({begin, matrix.rows});
	return batches;
}

static DmsaBatchSummary summarize_dmsa_batches(const PackedInput& packed,
		id_t rows) {
	constexpr id_t kMinRows = 4;
	constexpr id_t kMinProducts = 512;
	constexpr id_t kMaxRows = 64;
	// The 8x4 DMSA backend processes eight rows concurrently.  Within a PMCU,
	// feedback rounds accept three additional tasks after the first four, so
	// the U280 backend is bounded by ROW_NNZ_MAX=180 rather than 8*4 tasks.
	constexpr id_t kMaxTasks = 180;
	DmsaBatchSummary summary;
	id_t row = 0;
	while (row < rows) {
		const id_t tasks = packed.row_task_ptr[row + 1]
			- packed.row_task_ptr[row];
		const id_t mode = packed.route[row] & kRouteModeMask;
		const id_t products = packed.route[row] >> kRouteModeBits;
		if (mode != 2 || tasks < 2 || tasks > kMaxTasks || products == 0) {
			row++;
			continue;
		}
		id_t batch_rows = 0;
		uint64_t batch_products = 0;
		while (row + batch_rows < rows && batch_rows < kMaxRows) {
			const id_t candidate = row + batch_rows;
			const id_t candidate_tasks = packed.row_task_ptr[candidate + 1]
				- packed.row_task_ptr[candidate];
			const id_t candidate_mode = packed.route[candidate]
				& kRouteModeMask;
			const id_t candidate_products = packed.route[candidate]
				>> kRouteModeBits;
			if (candidate_mode != 2 || candidate_tasks < 2
					|| candidate_tasks > kMaxTasks
					|| candidate_products == 0)
				break;
			batch_rows++;
			batch_products += candidate_products;
		}
		if (batch_rows >= kMinRows && batch_products >= kMinProducts) {
			summary.selected_rows += batch_rows;
			summary.selected_products += batch_products;
			summary.batches++;
		} else {
			summary.fallback_rows += batch_rows;
		}
		row += batch_rows;
	}
	return summary;
}

struct DenseReaderCacheSummary {
	uint64_t dense_commands = 0;
	uint64_t cacheable_commands = 0;
	uint64_t cache_hits = 0;
	uint64_t dense_beats = 0;
	uint64_t saved_hbm_beats = 0;
	std::array<uint64_t, kShards> reader_commands{};
	std::array<uint64_t, kShards> reader_hits{};
};

// Reproduce the deployed DENSE-only reader cache from the already packed task
// stream.  This is a Host diagnostic only: it neither changes route tags nor
// contributes to selector time.  Following the dispatcher's aligned two-row
// interleave is important because each reader owns an independent 16-entry
// round-robin cache and therefore observes only its own command subsequence.
static DenseReaderCacheSummary summarize_dense_reader_cache(
		const PackedInput& packed, id_t rows, id_t columns) {
	constexpr unsigned kCacheRows = 16;
	constexpr unsigned kCacheRowBeats = 8;
	struct CacheEntry {
		id_t offset = 0;
		id_t beats = 0;
		bool valid = false;
	};
	std::array<std::array<CacheEntry, kCacheRows>, kShards> cache{};
	std::array<unsigned, kShards> replacement{};
	DenseReaderCacheSummary summary;

	auto accept_task = [&](id_t task, bool dense_mode) {
		const TaskWord512& word = packed.tasks[task >> 2];
		const uint64_t descriptor = word.lane[(task & 3U) * 2U];
		const id_t offset = static_cast<id_t>(descriptor);
		const id_t length = static_cast<id_t>((descriptor >> 32) & 0xffffffU);
		const unsigned shard = static_cast<unsigned>((descriptor >> 56) & 7U);
		const id_t beats = (length + 7) >> 3;
		if (!dense_mode || beats == 0) return;
		++summary.dense_commands;
		++summary.reader_commands[shard];
		summary.dense_beats += beats;
		if (beats > kCacheRowBeats) return;
		++summary.cacheable_commands;
		bool hit = false;
		for (const CacheEntry& entry : cache[shard])
			if (entry.valid && entry.offset == offset && entry.beats == beats) {
				hit = true;
				break;
			}
		if (hit) {
			++summary.cache_hits;
			++summary.reader_hits[shard];
			summary.saved_hbm_beats += beats;
		} else {
			CacheEntry& entry = cache[shard][replacement[shard]];
			entry = {offset, beats, true};
			replacement[shard] = (replacement[shard] + 1) & (kCacheRows - 1);
		}
	};

	for (id_t base_row = 0; base_row < rows; base_row += 2) {
		const id_t rows_in_group = base_row + 1 < rows ? 2 : 1;
		id_t next[2] = {packed.row_task_ptr[base_row], 0};
		id_t end[2] = {packed.row_task_ptr[base_row + 1], 0};
		id_t mode[2] = {packed.route[base_row] & kRouteModeMask, 0};
		if (rows_in_group == 2) {
			next[1] = packed.row_task_ptr[base_row + 1];
			end[1] = packed.row_task_ptr[base_row + 2];
			mode[1] = packed.route[base_row + 1] & kRouteModeMask;
		}
		const bool dense_pair = rows_in_group == 2 && columns <= 32768
			&& mode[0] == 3 && mode[1] == 3;
		if (dense_pair) {
			unsigned turn = 0;
			while (next[0] < end[0] || next[1] < end[1]) {
				const unsigned context = next[turn] < end[turn] ? turn : turn ^ 1U;
				accept_task(next[context]++, true);
				turn = context ^ 1U;
			}
		} else {
			for (id_t context = 0; context < rows_in_group; ++context)
				while (next[context] < end[context])
					accept_task(next[context]++, mode[context] == 3);
		}
	}
	return summary;
}

static void verify(const Reference& ref, const std::vector<id_t>& rowptr,
		const std::vector<uint64_t>& item, const std::vector<id_t>& stats) {
	if (stats[10] || stats[11])
		throw std::runtime_error("kernel input/output overflow: input_rows="
			+ std::to_string(stats[10]) + " output_rows="
			+ std::to_string(stats[11]));
	if (rowptr.size() != ref.rowptr.size() || rowptr.empty()
			|| rowptr[0] != 0 || rowptr.back() != stats[9]
			|| stats[9] > item.size())
		throw std::runtime_error("kernel returned malformed CSR row pointers: "
			+ std::string("rowptr_size=") + std::to_string(rowptr.size())
			+ " expected_size=" + std::to_string(ref.rowptr.size())
			+ " first=" + std::to_string(rowptr.empty() ? 0 : rowptr.front())
			+ " penultimate=" + std::to_string(rowptr.size() < 2
				? 0 : rowptr[rowptr.size() - 2])
			+ " last=" + std::to_string(rowptr.empty() ? 0 : rowptr.back())
			+ " expected_last_row_nnz=" + std::to_string(ref.rowptr.size() < 2
				? 0 : ref.rowptr.back() - ref.rowptr[ref.rowptr.size() - 2])
			+ " stats_output=" + std::to_string(stats[9])
			+ " capacity=" + std::to_string(item.size()));

	size_t structural_roundoff = 0;
	for (size_t row = 0; row + 1 < rowptr.size(); ++row) {
		if (rowptr[row] > rowptr[row + 1] || rowptr[row + 1] > stats[9])
			throw std::runtime_error("kernel returned invalid CSR row "
				+ std::to_string(row));
		size_t expected = ref.rowptr[row];
		const size_t expected_end = ref.rowptr[row + 1];
		size_t actual = rowptr[row];
		const size_t actual_end = rowptr[row + 1];
		id_t previous_col = 0;
		bool have_previous = false;
		while (expected < expected_end || actual < actual_end) {
			const id_t expected_col = expected < expected_end
				? ref.col[expected] : std::numeric_limits<id_t>::max();
			const id_t actual_col = actual < actual_end
				? static_cast<id_t>(item[actual]) : std::numeric_limits<id_t>::max();
			if (actual < actual_end
					&& (actual_col >= ref.rowptr.size() - 1
						|| (have_previous && actual_col <= previous_col)))
				throw std::runtime_error("kernel returned unsorted/out-of-range "
					"column in row " + std::to_string(row));
			double expected_value = 0.0;
			double expected_abs_sum = 0.0;
			id_t expected_terms = 0;
			double actual_value = 0.0;
			if (expected_col <= actual_col) {
				expected_value = ref.val[expected];
				expected_abs_sum = ref.abs_sum[expected];
				expected_terms = ref.terms[expected];
			}
			if (actual_col <= expected_col)
				actual_value = bits_float(
					static_cast<uint32_t>(item[actual] >> 32));
			// A dot product of p FP32 terms performs p rounded multiplies and
			// roughly p rounded additions.  gamma_(2p+8) bounds those legal
			// reorderings against the double-precision reference.  Retain a
			// small mixed absolute/relative floor for short, well-conditioned
			// rows.
			const double unit_roundoff = std::ldexp(1.0, -24);
			const double operations = 2.0 * expected_terms + 8.0;
			const double scaled_roundoff = operations * unit_roundoff;
			const double gamma = scaled_roundoff < 1.0
				? scaled_roundoff / (1.0 - scaled_roundoff)
				: std::numeric_limits<double>::infinity();
			const double tolerance = std::max(
				1.0e-4 + 2.0e-4 * std::fabs(expected_value),
				gamma * expected_abs_sum);
			if (!std::isfinite(actual_value)
					|| std::fabs(actual_value - expected_value) > tolerance)
				throw std::runtime_error("kernel value mismatch row="
					+ std::to_string(row)
					+ " col=" + std::to_string(
						std::min(expected_col, actual_col))
					+ " value=" + std::to_string(actual_value) + "/"
					+ std::to_string(expected_value)
					+ " abs_diff=" + std::to_string(
						std::fabs(actual_value - expected_value))
					+ " tolerance=" + std::to_string(tolerance)
					+ " terms=" + std::to_string(expected_terms)
					+ " abs_sum=" + std::to_string(expected_abs_sum));
			if (expected_col != actual_col) ++structural_roundoff;
			if (expected_col <= actual_col) ++expected;
			if (actual_col <= expected_col) {
				previous_col = actual_col;
				have_previous = true;
				++actual;
			}
		}
	}
	if (structural_roundoff != 0)
		std::cout << "verification_structural_roundoff="
			<< structural_roundoff << ' ';
}

// Validate one scalable (row batch, column window) without constructing the
// complete A^2 on the Host.  Every CSR boundary is checked.  Small tiles are
// checked row-for-row; large tiles use a deterministic, evenly-spaced sample
// plus the first/last rows.  This keeps validation memory bounded while still
// comparing the selected rows against an independent double-precision sum.
static std::vector<RowStructureFingerprint> fingerprint_output_rows(
		const std::vector<id_t>& rowptr,
		const std::array<const Beat512*, 4>& C_bank) {
	const auto get_item = [&](uint64_t index) -> uint64_t {
		const uint64_t word = index >> 3;
		return C_bank[word & 3U][word >> 2].lane[index & 7U];
	};
	std::vector<RowStructureFingerprint> result(rowptr.size() - 1);
	for (size_t row = 0; row + 1 < rowptr.size(); ++row) {
		uint64_t first = 1469598103934665603ULL;
		uint64_t second = 0x9e3779b97f4a7c15ULL;
		for (uint64_t index = rowptr[row]; index < rowptr[row + 1]; ++index) {
			const uint64_t column = static_cast<uint32_t>(get_item(index));
			first = (first ^ column) * 1099511628211ULL;
			uint64_t mixed = column + 0x9e3779b97f4a7c15ULL
				+ ((index - rowptr[row]) << 1);
			mixed = (mixed ^ (mixed >> 30)) * 0xbf58476d1ce4e5b9ULL;
			mixed = (mixed ^ (mixed >> 27)) * 0x94d049bb133111ebULL;
			second ^= mixed ^ (mixed >> 31);
		}
		result[row] = {first, second};
	}
	return result;
}

static uint64_t verify_window_tile(const Csr& matrix,
		const WindowPackResult& tile, uint32_t window_width,
		const std::vector<id_t>& rowptr,
		const std::array<const Beat512*, 4>& C_bank,
		const std::vector<id_t>& stats,
		const std::vector<id_t>& structural_crosscheck_rows,
		const std::vector<uint64_t>* logical_items = nullptr,
		bool include_standard_sample = true) {
	if (stats.size() < kStats || stats[10] != 0 || stats[11] != 0)
		throw std::runtime_error("scalable kernel input/output overflow");
	const id_t rows = tile.row_end - tile.row_begin;
	if (rowptr.size() != uint64_t(rows) + 1 || rowptr.empty()
			|| rowptr.front() != 0 || rowptr.back() != stats[9]
			|| rowptr.back() > tile.output_nnz_upper_bound)
		throw std::runtime_error("scalable kernel returned malformed CSR bounds");
	for (id_t row = 0; row < rows; ++row)
		if (rowptr[row] > rowptr[row + 1] || rowptr[row + 1] > stats[9])
			throw std::runtime_error("scalable kernel returned invalid row pointer");

	const auto get_item = [&](uint64_t index) -> uint64_t {
		if (logical_items != nullptr) return logical_items->at(index);
		const uint64_t word = index >> 3;
		return C_bank[word & 3U][word >> 2].lane[index & 7U];
	};
	// Exact checking is inexpensive below this product threshold.  Above it,
	// 512 evenly-spaced rows give a fixed validation cost independent of N.
	constexpr uint64_t kExactProducts = 10000000;
	constexpr id_t kSampleRows = 512;
	std::vector<id_t> check_rows;
	if (!include_standard_sample) {
		// The baseline already received the standard sample when it ran.  A later
		// mode only adds newly discovered cross-mode structural differences here.
	} else if (tile.products <= kExactProducts || rows <= kSampleRows) {
		check_rows.resize(rows);
		std::iota(check_rows.begin(), check_rows.end(), 0);
	} else {
		check_rows.reserve(kSampleRows + 2);
		for (id_t sample = 0; sample < kSampleRows; ++sample)
			check_rows.push_back(static_cast<id_t>(
				(uint64_t(sample) * (rows - 1)) / (kSampleRows - 1)));
		std::sort(check_rows.begin(), check_rows.end());
		check_rows.erase(std::unique(check_rows.begin(), check_rows.end()),
			check_rows.end());
	}
	check_rows.insert(check_rows.end(), structural_crosscheck_rows.begin(),
		structural_crosscheck_rows.end());
	std::sort(check_rows.begin(), check_rows.end());
	check_rows.erase(std::unique(check_rows.begin(), check_rows.end()),
		check_rows.end());

	const uint64_t column_base = uint64_t(tile.window) * kWindowColumns;
	const uint64_t column_end = column_base + window_width;
	for (id_t local_row : check_rows) {
		struct Accum {
			double value = 0.0;
			double abs_sum = 0.0;
			id_t terms = 0;
		};
		std::map<id_t, Accum> expected;
		const id_t row = tile.row_begin + local_row;
		for (id_t at = matrix.rowptr[row]; at < matrix.rowptr[row + 1]; ++at) {
			const id_t k = matrix.col[at];
			const auto begin = std::lower_bound(
				matrix.col.begin() + matrix.rowptr[k],
				matrix.col.begin() + matrix.rowptr[k + 1],
				static_cast<id_t>(column_base));
			const auto end = std::lower_bound(begin,
				matrix.col.begin() + matrix.rowptr[k + 1],
				static_cast<id_t>(column_end));
			for (auto item = begin; item != end; ++item) {
				const id_t index = static_cast<id_t>(item - matrix.col.begin());
				const double product = static_cast<double>(matrix.val[at])
					* static_cast<double>(matrix.val[index]);
				Accum& accum = expected[*item - column_base];
				accum.value += product;
				accum.abs_sum += std::fabs(product);
				accum.terms++;
			}
		}

		auto expected_it = expected.begin();
		uint64_t actual = rowptr[local_row];
		const uint64_t actual_end = rowptr[local_row + 1];
		id_t previous_col = 0;
		bool have_previous = false;
		while (expected_it != expected.end() || actual < actual_end) {
			const id_t expected_col = expected_it != expected.end()
				? expected_it->first : std::numeric_limits<id_t>::max();
			const uint64_t raw = actual < actual_end ? get_item(actual) : 0;
			const id_t actual_col = actual < actual_end
				? static_cast<id_t>(raw) : std::numeric_limits<id_t>::max();
			if (actual < actual_end && (actual_col >= window_width
					|| (have_previous && actual_col <= previous_col)))
				throw std::runtime_error("scalable output column order mismatch row="
					+ std::to_string(row));
			double expected_value = 0.0;
			double abs_sum = 0.0;
			id_t terms = 0;
			double actual_value = 0.0;
			if (expected_col <= actual_col) {
				expected_value = expected_it->second.value;
				abs_sum = expected_it->second.abs_sum;
				terms = expected_it->second.terms;
			}
			if (actual_col <= expected_col)
				actual_value = bits_float(static_cast<uint32_t>(raw >> 32));
			const double scaled_roundoff = (2.0 * terms + 8.0)
				* std::ldexp(1.0, -24);
			const double gamma = scaled_roundoff < 1.0
				? scaled_roundoff / (1.0 - scaled_roundoff)
				: std::numeric_limits<double>::infinity();
			const double tolerance = std::max(
				1.0e-4 + 2.0e-4 * std::fabs(expected_value),
				gamma * abs_sum);
			if (!std::isfinite(actual_value)
					|| std::fabs(actual_value - expected_value) > tolerance)
				throw std::runtime_error("scalable value mismatch row="
					+ std::to_string(row) + " local_col="
					+ std::to_string(std::min(expected_col, actual_col)));
			if (expected_col <= actual_col) ++expected_it;
			if (actual_col <= expected_col) {
				previous_col = actual_col;
				have_previous = true;
				++actual;
			}
		}
	}
	return check_rows.size();
}

static double median(std::vector<double> values) {
	std::sort(values.begin(), values.end());
	const size_t middle = values.size() / 2;
	if (values.size() & 1U) return values[middle];
	return 0.5 * (values[middle - 1] + values[middle]);
}

static const char* global_mode_name(adaptive_host::GlobalMode mode) {
	switch (mode) {
	case adaptive_host::DENSE_ONLY: return "dense_only";
	case adaptive_host::ADAPTIVE: return "row_adaptive";
	default: return "merge_preferred_capacity_safe";
	}
}

static ForceMode force_mode_for(adaptive_host::GlobalMode mode) {
	if (mode == adaptive_host::DENSE_ONLY) return kForceDense;
	if (mode == adaptive_host::ADAPTIVE) return kUseRoute;
	return kForceMerge;
}

// The DENSE dispatcher consumes one row through all eight shard readers.  A
// very large batch can make the reference-weighted B placement pathological:
// one high-work DENSE row can then own hundreds of commands on busy shards but
// none on one or more readers.  The deployed kernel cannot safely retire that
// command shape.  Repacking a smaller row interval changes the placement weights
// and restores progress without changing the matrix, route, or arithmetic.
//
// Return an interior row boundary that isolates the first risky row.  When that
// row is already at the right boundary, split before it; a one-row tile is left
// alone because it cannot be split further and its own repack is the fallback.
static bool find_dense_empty_shard_split(const WindowPackResult& tile,
		bool force_all_nonempty_dense, id_t* split_row) {
	constexpr id_t kLargeDenseTasks = 512;
	const PackedInput& packed = tile.packed;
	const id_t tile_rows = tile.row_end - tile.row_begin;
	for (id_t local_row = 0; local_row < tile_rows; ++local_row) {
		const id_t route = packed.route[local_row] & kRouteModeMask;
		if (force_all_nonempty_dense ? route == 0U : route != 3U) continue;
		const id_t task_begin = packed.row_task_ptr[local_row];
		const id_t task_end = packed.row_task_ptr[local_row + 1];
		if (task_end - task_begin < kLargeDenseTasks) continue;
		std::array<id_t, kShards> shard_tasks{};
		for (id_t task = task_begin; task < task_end; ++task) {
			const TaskWord512& word = packed.tasks[task >> 2];
			const uint64_t descriptor = word.lane[(task & 3U) * 2U];
			++shard_tasks[(descriptor >> 56) & 7U];
		}
		if (std::none_of(shard_tasks.begin(), shard_tasks.end(),
				[](id_t tasks) { return tasks == 0; }))
			continue;
		if (local_row + 1 < tile_rows) {
			*split_row = tile.row_begin + local_row + 1;
			return true;
		}
		if (local_row != 0) {
			*split_row = tile.row_begin + local_row;
			return true;
		}
	}
	return false;
}

static ScalableExecutionResult execute_scalable_windows(const Csr& matrix,
		const ScalableIndex& index, const std::string& xclbin_path,
		const std::string& device_arg, const std::string& csv_path,
		unsigned reps) {
	if (lower(device_arg) == "auto" && std::getenv("TARGET_CARD") == nullptr)
		::setenv("TARGET_CARD", "u280", 0);
	xrt::device device(host_device::parse_device_arg(device_arg));
	const auto uuid = device.load_xclbin(xclbin_path);
	const bool launch_trace = std::getenv("ADAPT_LAUNCH_TRACE") != nullptr;
	if (launch_trace)
		std::cerr << "launch_trace stage=xclbin_load_complete" << std::endl;
	// XRT 2022.2 can retain an exclusive CU context after an interrupted Host
	// process.  Shared is also the XRT API default and is sufficient here because
	// every run object still targets the single programmed CU.  Keep an explicit
	// diagnostic escape hatch for administrators testing exclusive ownership.
	const xrt::kernel::cu_access_mode access_mode
		= std::getenv("ADAPT_EXCLUSIVE_CU") != nullptr
		? xrt::kernel::cu_access_mode::exclusive
		: xrt::kernel::cu_access_mode::shared;
	xrt::kernel kernel(device, uuid,
		"adaptive_hbm_spgemm:{adaptive_hbm_spgemm_1}",
		access_mode);
	if (launch_trace)
		std::cerr << "launch_trace stage=kernel_open_complete" << std::endl;
	std::ofstream csv(csv_path);
	if (!csv) throw std::runtime_error("cannot open scalable result CSV");
	csv << "matrix,mode,window,batch,row_begin,row_end,reps,packing_ms,h2d_ms,"
		"kernel_ms,d2h_ms,verification_ms,products,output_nnz,merge_rows,"
		"dense_rows,sampled_rows,verified,default_merge_rows,"
		"default_dense_rows,default_merge_products,default_dense_products,"
		"capacity_forced_dense_rows,conservative_capacity_locked_rows,"
		"structural_crosscheck_rows,baseline_crosscheck_rows\n";

	std::vector<std::pair<std::string, ForceMode>> modes = {
		{"row_adaptive", kUseRoute},
		{"merge_preferred_capacity_safe", kForceMerge},
		{"dense_only", kForceDense}};
	if (const char* requested_mode = std::getenv("ADAPT_SCALABLE_MODE")) {
		const std::string requested = lower(requested_mode);
		if (requested == "all") {
			// Keep the default three-mode campaign.
		} else if (requested == "adaptive" || requested == "row_adaptive")
			modes = {{"row_adaptive", kUseRoute}};
		else if (requested == "merge" || requested == "merge_only")
			modes = {{"merge_preferred_capacity_safe", kForceMerge}};
		else if (requested == "dense" || requested == "dense_only")
			modes = {{"dense_only", kForceDense}};
		else if (requested != "all")
			throw std::runtime_error("invalid ADAPT_SCALABLE_MODE: " + requested);
	}
	std::vector<ScalableExecutionResult> totals(modes.size());
	const bool includes_dense_only = std::any_of(modes.begin(), modes.end(),
		[](const auto& mode) { return mode.second == kForceDense; });
	std::vector<std::vector<double>> whole_kernel_samples(
		modes.size(), std::vector<double>(reps, 0.0));
	id_t diagnostic_row_begin = 0;
	id_t diagnostic_row_end = matrix.rows;
	bool diagnostic_row_range = false;
	if (const char* requested_range = std::getenv(
			"ADAPT_SCALABLE_ROW_RANGE")) {
		char* separator = nullptr;
		const unsigned long begin = std::strtoul(requested_range,
			&separator, 10);
		if (separator == requested_range || *separator != ':')
			throw std::runtime_error(
				"ADAPT_SCALABLE_ROW_RANGE must be begin:end");
		char* end_pointer = nullptr;
		const unsigned long end = std::strtoul(separator + 1,
			&end_pointer, 10);
		if (end_pointer == separator + 1 || *end_pointer != '\0'
				|| begin >= end || end > matrix.rows)
			throw std::runtime_error(
				"ADAPT_SCALABLE_ROW_RANGE is outside the matrix");
		diagnostic_row_begin = static_cast<id_t>(begin);
		diagnostic_row_end = static_cast<id_t>(end);
		diagnostic_row_range = true;
		std::cerr << "diagnostic_row_range=[" << diagnostic_row_begin
			<< ',' << diagnostic_row_end << ")" << std::endl;
	}
	const ScalableWindowSummary& summary = index.summary;
	uint32_t diagnostic_window_begin = 0;
	uint32_t diagnostic_window_end = summary.windows;
	if (const char* requested_range = std::getenv(
			"ADAPT_SCALABLE_WINDOW_RANGE")) {
		char* separator = nullptr;
		const unsigned long begin = std::strtoul(requested_range,
			&separator, 10);
		if (separator == requested_range || *separator != ':')
			throw std::runtime_error(
				"ADAPT_SCALABLE_WINDOW_RANGE must be begin:end");
		char* end_pointer = nullptr;
		const unsigned long end = std::strtoul(separator + 1,
			&end_pointer, 10);
		if (end_pointer == separator + 1 || *end_pointer != '\0'
				|| begin >= end || end > summary.windows)
			throw std::runtime_error(
				"ADAPT_SCALABLE_WINDOW_RANGE is outside the matrix");
		diagnostic_window_begin = static_cast<uint32_t>(begin);
		diagnostic_window_end = static_cast<uint32_t>(end);
		std::cerr << "diagnostic_window_range=[" << diagnostic_window_begin
			<< ',' << diagnostic_window_end << ")" << std::endl;
	}
	for (uint32_t window = diagnostic_window_begin;
			window < diagnostic_window_end; ++window) {
		const uint32_t window_width = static_cast<uint32_t>(std::min<uint64_t>(
			kWindowColumns,
			uint64_t(matrix.cols) - uint64_t(window) * kWindowColumns));
		auto batches = make_window_row_batches(matrix, index, window);
		if (diagnostic_row_range)
			batches = {{diagnostic_row_begin, diagnostic_row_end}};
		for (size_t batch = 0; batch < batches.size();) {
			const auto pack_begin = Clock::now();
			const WindowPackResult tile = pack_window_tile(matrix, index, window,
				batches[batch].first, batches[batch].second);
			id_t split_row = 0;
			if (!diagnostic_row_range
					&& find_dense_empty_shard_split(
						tile, includes_dense_only, &split_row)) {
				const auto original = batches[batch];
				if (split_row <= original.first || split_row >= original.second)
					throw std::runtime_error(
						"internal scalable repack split is not interior");
				batches[batch] = {original.first, split_row};
				batches.insert(batches.begin() + batch + 1,
					{split_row, original.second});
				std::cerr << "scalable_repack_split window=" << window
					<< " batch=" << batch << " rows=[" << original.first << ','
					<< original.second << ") split=" << split_row
					<< " reason=large_dense_empty_shard\n";
				continue;
			}
			if (launch_trace)
				std::cerr << "launch_trace stage=tile_pack_complete window="
					<< window << " batch=" << batch << std::endl;
			const double packing_ms = std::chrono::duration<double, std::milli>(
				Clock::now() - pack_begin).count();
			const id_t tile_rows = tile.row_end - tile.row_begin;
			const id_t capacity = static_cast<id_t>(std::max<uint64_t>(
				1, tile.output_nnz_upper_bound));
			const uint64_t capacity_words = (uint64_t(capacity) + 7) / 8;
			const size_t words_per_bank = static_cast<size_t>(std::max<uint64_t>(
				1, (capacity_words + 3) / 4));
			// Fail with an actionable capacity diagnostic before asking XRT to
			// allocate a bank.  XRT 2022.2 otherwise reports an over-capacity BO as
			// the misleading and context-free `std::bad_alloc`.
			constexpr uint64_t kPhysicalHbmBankBytes = 256ULL << 20;
			const auto require_bank_capacity = [&](const std::string& label,
					uint64_t bytes) {
				if (bytes <= kPhysicalHbmBankBytes) return;
				std::ostringstream message;
				message << "scalable " << label << " BO exceeds one 256-MiB HBM bank"
					<< " window=" << window << " batch=" << batch
					<< " rows=[" << tile.row_begin << ',' << tile.row_end << ')'
					<< " bytes=" << bytes;
				throw std::runtime_error(message.str());
			};
			require_bank_capacity("task",
				uint64_t(tile.packed.tasks.size()) * sizeof(TaskWord512));
			require_bank_capacity("row",
				uint64_t(tile.packed.row_task_ptr.size()) * sizeof(id_t));
			require_bank_capacity("route",
				uint64_t(tile.packed.route.size()) * sizeof(id_t));
			for (unsigned shard = 0; shard < kShards; ++shard)
				require_bank_capacity("B" + std::to_string(shard),
					uint64_t(tile.packed.B[shard].size()) * sizeof(Beat512));
			require_bank_capacity("C-row",
				(uint64_t(tile_rows) + 1) * sizeof(id_t));
			require_bank_capacity("C-item",
				uint64_t(words_per_bank) * sizeof(Beat512));

			xrt::bo task_bo(device,
				tile.packed.tasks.size() * sizeof(TaskWord512), kernel.group_id(0));
			xrt::bo row_bo(device,
				tile.packed.row_task_ptr.size() * sizeof(id_t), kernel.group_id(1));
			xrt::bo route_bo(device,
				tile.packed.route.size() * sizeof(id_t), kernel.group_id(2));
			std::vector<xrt::bo> B_bo;
			for (unsigned shard = 0; shard < kShards; ++shard)
				B_bo.emplace_back(device,
					tile.packed.B[shard].size() * sizeof(Beat512),
					kernel.group_id(3 + shard));
			xrt::bo C_row_bo(device, (uint64_t(tile_rows) + 1) * sizeof(id_t),
				kernel.group_id(23));
			std::array<xrt::bo, 4> C_item_bo = {
				xrt::bo(device, words_per_bank * sizeof(Beat512), kernel.group_id(24)),
				xrt::bo(device, words_per_bank * sizeof(Beat512), kernel.group_id(25)),
				xrt::bo(device, words_per_bank * sizeof(Beat512), kernel.group_id(26)),
				xrt::bo(device, words_per_bank * sizeof(Beat512), kernel.group_id(27))};
			xrt::bo stats_bo(device, kStats * sizeof(id_t), kernel.group_id(28));
			if (launch_trace)
				std::cerr << "launch_trace stage=bo_allocation_complete window="
					<< window << " batch=" << batch << std::endl;

			std::memcpy(task_bo.map<void*>(), tile.packed.tasks.data(),
				tile.packed.tasks.size() * sizeof(TaskWord512));
			std::memcpy(row_bo.map<void*>(), tile.packed.row_task_ptr.data(),
				tile.packed.row_task_ptr.size() * sizeof(id_t));
			std::memcpy(route_bo.map<void*>(), tile.packed.route.data(),
				tile.packed.route.size() * sizeof(id_t));
			for (unsigned shard = 0; shard < kShards; ++shard)
				std::memcpy(B_bo[shard].map<void*>(), tile.packed.B[shard].data(),
					tile.packed.B[shard].size() * sizeof(Beat512));
			std::memset(stats_bo.map<void*>(), 0, kStats * sizeof(id_t));
			const auto h2d_begin = Clock::now();
			task_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
			row_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
			route_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
			for (auto& bo : B_bo) bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
			stats_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
			if (launch_trace)
				std::cerr << "launch_trace stage=h2d_complete window=" << window
					<< " batch=" << batch << std::endl;
			const double h2d_ms = std::chrono::duration<double, std::milli>(
				Clock::now() - h2d_begin).count();

			const auto launch = [&](ForceMode force_mode) {
				return kernel(task_bo, row_bo, route_bo,
					B_bo[0], B_bo[1], B_bo[2], B_bo[3],
					B_bo[4], B_bo[5], B_bo[6], B_bo[7],
					static_cast<id_t>(tile.packed.B[0].size()),
					static_cast<id_t>(tile.packed.B[1].size()),
					static_cast<id_t>(tile.packed.B[2].size()),
					static_cast<id_t>(tile.packed.B[3].size()),
					static_cast<id_t>(tile.packed.B[4].size()),
					static_cast<id_t>(tile.packed.B[5].size()),
					static_cast<id_t>(tile.packed.B[6].size()),
					static_cast<id_t>(tile.packed.B[7].size()),
					tile_rows, static_cast<id_t>(window_width),
					static_cast<uint32_t>(force_mode), capacity, C_row_bo,
					C_item_bo[0], C_item_bo[1], C_item_bo[2], C_item_bo[3],
					stats_bo);
			};
			std::vector<id_t> baseline_rowptr;
			std::vector<RowStructureFingerprint> baseline_structure;
			std::vector<id_t> baseline_stats;
			std::vector<uint64_t> baseline_items;
			std::vector<uint8_t> baseline_crosschecked(tile_rows, 0);
			for (size_t mode_index = 0; mode_index < modes.size(); ++mode_index) {
				if (launch_trace)
					std::cerr << "launch_trace stage=warmup_submit_begin mode="
						<< modes[mode_index].first << std::endl;
				auto warmup = launch(modes[mode_index].second);
				if (launch_trace)
					std::cerr << "launch_trace stage=warmup_submit_complete mode="
						<< modes[mode_index].first << std::endl;
				warmup.wait();
				if (launch_trace)
					std::cerr << "launch_trace stage=warmup_wait_complete mode="
						<< modes[mode_index].first << std::endl;
				std::vector<double> times;
				for (unsigned rep = 0; rep < reps; ++rep) {
					const auto begin = Clock::now();
					auto run = launch(modes[mode_index].second);
					run.wait();
					times.push_back(std::chrono::duration<double, std::milli>(
						Clock::now() - begin).count());
				}
				const double kernel_ms = median(times);
				for (unsigned rep = 0; rep < reps; ++rep)
					whole_kernel_samples[mode_index][rep] += times[rep];

				const auto d2h_begin = Clock::now();
				C_row_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
				stats_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
				std::vector<id_t> rowptr(uint64_t(tile_rows) + 1);
				std::vector<id_t> stats(kStats);
				std::memcpy(rowptr.data(), C_row_bo.map<void*>(),
					rowptr.size() * sizeof(id_t));
				std::memcpy(stats.data(), stats_bo.map<void*>(),
					stats.size() * sizeof(id_t));
				if (stats[9] > capacity)
					throw std::runtime_error(
						"scalable output exceeds allocated capacity");
				const uint64_t actual_words = (uint64_t(stats[9]) + 7) / 8;
				for (unsigned bank = 0; bank < 4; ++bank) {
					const uint64_t bank_words = actual_words <= bank ? 0
						: (actual_words - bank + 3) / 4;
					if (bank_words != 0)
						C_item_bo[bank].sync(XCL_BO_SYNC_BO_FROM_DEVICE,
							bank_words * sizeof(Beat512), 0);
				}
				const double d2h_ms = std::chrono::duration<double, std::milli>(
					Clock::now() - d2h_begin).count();
				const std::array<const Beat512*, 4> C_bank = {
					C_item_bo[0].map<const Beat512*>(),
					C_item_bo[1].map<const Beat512*>(),
					C_item_bo[2].map<const Beat512*>(),
					C_item_bo[3].map<const Beat512*>()};
				const auto get_current_item = [&](uint64_t index) -> uint64_t {
					const uint64_t word = index >> 3;
					return C_bank[word & 3U][word >> 2].lane[index & 7U];
				};
				const auto structure = fingerprint_output_rows(rowptr, C_bank);
				std::vector<id_t> structural_crosscheck_rows;
				std::vector<id_t> new_baseline_crosscheck_rows;
				if (mode_index == 0) {
					baseline_rowptr = rowptr;
					baseline_structure = structure;
					baseline_stats = stats;
					baseline_items.resize(stats[9]);
					for (uint64_t index = 0; index < stats[9]; ++index)
						baseline_items[index] = get_current_item(index);
				} else {
					for (id_t local_row = 0; local_row < tile_rows; ++local_row) {
						const bool different_length =
							rowptr[local_row + 1] - rowptr[local_row]
							!= baseline_rowptr[local_row + 1]
								- baseline_rowptr[local_row];
						const bool different_columns =
							structure[local_row].first
								!= baseline_structure[local_row].first
							|| structure[local_row].second
								!= baseline_structure[local_row].second;
						if (different_length || different_columns) {
							structural_crosscheck_rows.push_back(local_row);
							if (!baseline_crosschecked[local_row]) {
								baseline_crosschecked[local_row] = 1;
								new_baseline_crosscheck_rows.push_back(local_row);
							}
						}
					}
				}
				const auto verify_begin = Clock::now();
				const uint64_t sampled = verify_window_tile(matrix, tile,
					window_width, rowptr, C_bank, stats,
					structural_crosscheck_rows);
				if (!new_baseline_crosscheck_rows.empty())
					verify_window_tile(matrix, tile, window_width,
						baseline_rowptr, C_bank, baseline_stats,
						new_baseline_crosscheck_rows, &baseline_items, false);
				const double verification_ms =
					std::chrono::duration<double, std::milli>(
						Clock::now() - verify_begin).count();

				ScalableExecutionResult& total = totals[mode_index];
				++total.tiles;
				total.packing_ms += packing_ms;
				total.h2d_ms += h2d_ms;
				total.d2h_ms += d2h_ms;
				total.verification_ms += verification_ms;
				total.output_nnz += stats[9];
				total.merge_rows += stats[3];
				total.dense_rows += stats[4];
				total.sampled_rows += sampled;
				total.default_merge_rows += tile.plan.merge_rows;
				total.default_dense_rows += tile.plan.dense_rows;
				total.default_merge_products += tile.plan.merge_products;
				total.default_dense_products += tile.plan.dense_products;
				total.capacity_forced_dense_rows
					+= tile.capacity_forced_dense_rows;
				total.conservative_capacity_locked_rows
					+= tile.conservative_capacity_locked_rows;
				total.structural_crosscheck_rows
					+= structural_crosscheck_rows.size();
				total.baseline_crosscheck_rows
					+= new_baseline_crosscheck_rows.size();
				std::cout << "scalable_region mode=" << modes[mode_index].first
					<< " window=" << window << " batch=" << batch << " rows=["
					<< tile.row_begin << ',' << tile.row_end << ") kernel_ms="
					<< kernel_ms << " output_nnz=" << stats[9]
					<< " route(merge,dense)=(" << stats[3] << ',' << stats[4] << ')'
					<< " default_route(merge,dense)=("
					<< tile.plan.merge_rows << ',' << tile.plan.dense_rows << ')'
					<< " default_route_products(merge,dense)=("
					<< tile.plan.merge_products << ','
					<< tile.plan.dense_products << ')'
					<< " capacity_locked(exact,conservative)=("
					<< tile.capacity_forced_dense_rows << ','
					<< tile.conservative_capacity_locked_rows << ')'
					<< " structural_crosscheck_rows="
					<< structural_crosscheck_rows.size()
					<< " baseline_crosscheck_rows="
					<< new_baseline_crosscheck_rows.size()
					<< " sampled_rows=" << sampled << " PASS\n";
				csv << "scalable," << modes[mode_index].first << ',' << window
					<< ',' << batch << ',' << tile.row_begin << ',' << tile.row_end
					<< ',' << reps << ',' << std::fixed << std::setprecision(6)
					<< packing_ms << ',' << h2d_ms << ',' << kernel_ms << ','
					<< d2h_ms << ',' << verification_ms << ',' << tile.products << ','
					<< stats[9] << ',' << stats[3] << ',' << stats[4] << ','
					<< sampled << ",1," << tile.plan.merge_rows << ','
					<< tile.plan.dense_rows << ',' << tile.plan.merge_products << ','
					<< tile.plan.dense_products << ','
					<< tile.capacity_forced_dense_rows << ','
					<< tile.conservative_capacity_locked_rows << ','
					<< structural_crosscheck_rows.size() << ','
					<< new_baseline_crosscheck_rows.size() << '\n';
			}
			++batch;
		}
	}
	// A scalable matrix is one logical operation made of several kernel
	// launches. Pair equal-numbered repetitions across all execution regions and take the
	// median of those complete-operation sums; do not report a sum of unrelated
	// per-region medians as the final kernel time.
	for (size_t mode_index = 0; mode_index < modes.size(); ++mode_index) {
		ScalableExecutionResult& total = totals[mode_index];
		total.kernel_ms = median(whole_kernel_samples[mode_index]);
		std::cout << "scalable_whole mode=" << modes[mode_index].first
			<< " kernel_samples_ms=[";
		for (size_t index = 0;
				index < whole_kernel_samples[mode_index].size(); ++index) {
			if (index != 0) std::cout << ',';
			std::cout << std::setprecision(9)
				<< whole_kernel_samples[mode_index][index];
		}
		std::cout << "] median_ms=" << total.kernel_ms << '\n';
		csv << "TOTAL," << modes[mode_index].first << ",-1,-1,0,"
			<< matrix.rows << ',' << reps << ',' << std::fixed
			<< std::setprecision(6) << total.packing_ms << ',' << total.h2d_ms
			<< ',' << total.kernel_ms << ',' << total.d2h_ms << ','
			<< total.verification_ms << ',' << summary.total_products << ','
			<< total.output_nnz << ',' << total.merge_rows << ','
			<< total.dense_rows << ',' << total.sampled_rows << ",1,"
			<< total.default_merge_rows << ',' << total.default_dense_rows << ','
			<< total.default_merge_products << ','
			<< total.default_dense_products << ','
			<< total.capacity_forced_dense_rows << ','
			<< total.conservative_capacity_locked_rows << ','
			<< total.structural_crosscheck_rows << ','
			<< total.baseline_crosscheck_rows << '\n';
	}
	return totals.front();
}

}  // namespace

int main(int argc, char** argv) {
	try {
		if (argc < 5 || argc > 6) {
			std::cerr << "usage: " << argv[0]
				<< " xclbin device matrix.mtx results.csv [reps]\n";
			return 2;
		}
		const std::string xclbin_path = argv[1];
		const bool dry_run = lower(argv[2]) == "dry-run";
		const unsigned reps = argc == 6 ? std::stoul(argv[5]) : 3;
		if (reps == 0) throw std::runtime_error("reps must be positive");
		const auto matrix_load_begin = Clock::now();
		const Csr matrix = load_matrix_market(argv[3]);
		const double matrix_load_ms = std::chrono::duration<double, std::milli>(
			Clock::now() - matrix_load_begin).count();
		// Use one capacity-parameterized execution model for every matrix.  A
		// narrow matrix naturally has one column window; a wider one has as many
		// as required.  The old single-window Host remains only as an explicit
		// regression escape hatch and is never selected from the matrix size.
		const bool legacy_single_window =
			std::getenv("ADAPT_SCALABLE_LEGACY_SINGLE_WINDOW") != nullptr;
		if (!legacy_single_window) {
			const auto window_begin = Clock::now();
			const ScalableIndex scalable_index = build_scalable_index(matrix);
			const ScalableWindowSummary& window = scalable_index.summary;
			const double window_ms = std::chrono::duration<double, std::milli>(
				Clock::now() - window_begin).count();
			std::cout << "execution_model=capacity_parameterized_windows"
				<< " window_capacity=" << kWindowColumns
				<< " scalable_matrix=" << matrix.rows << 'x' << matrix.cols
				<< " nnz=" << matrix.col.size()
				<< " windows=" << window.windows
				<< " total_window_tasks=" << window.total_window_tasks
				<< " products=" << window.total_products
				<< " output_nnz_upper=" << window.output_nnz_upper_bound
				<< " max_window_tasks=" << window.max_window_tasks
				<< " max_window_products=" << window.max_window_products
				<< " max_window_output_upper="
				<< window.max_window_output_upper_bound
				<< " max_b_beats_per_shard=" << window.max_b_beats_per_shard
				<< " max_row_window_runs=" << window.max_row_window_runs
				<< " max_row_window_products="
				<< window.max_row_window_products
				<< " minimum_execution_regions=" << window.minimum_output_tiles
				<< " matrix_load_ms=" << matrix_load_ms
				<< " window_analysis_ms=" << window_ms << '\n';
			if (dry_run && std::getenv("ADAPT_WINDOW_SUMMARY_ONLY") != nullptr) {
				std::cout << "SCALABLE_WINDOW_SUMMARY_ONLY: PASS\n";
				return 0;
			}
			if (dry_run) {
				for (uint32_t index = 0; index < window.windows; ++index) {
					const auto batches = make_window_row_batches(
						matrix, scalable_index, index);
					for (size_t batch = 0; batch < batches.size(); ++batch) {
						const auto pack_begin = Clock::now();
						const WindowPackResult tile = pack_window_tile(
							matrix, scalable_index, index,
							batches[batch].first, batches[batch].second);
						const double pack_ms = std::chrono::duration<double, std::milli>(
							Clock::now() - pack_begin).count();
						uint64_t max_shard_beats = 0;
						for (const auto& shard : tile.packed.B)
							max_shard_beats = std::max<uint64_t>(
								max_shard_beats, shard.size());
						std::cout << "window=" << index << " batch=" << batch
							<< " rows=[" << tile.row_begin << ',' << tile.row_end << ')'
							<< " tasks=" << tile.packed.row_task_ptr.back()
							<< " products=" << tile.products
							<< " output_nnz_upper=" << tile.output_nnz_upper_bound
							<< " route(merge,dense)=(" << tile.plan.merge_rows << ','
							<< tile.plan.dense_rows << ')'
							<< " empty=" << tile.plan.empty_rows
							<< " direct=" << tile.plan.direct_rows
							<< " capacity_forced_dense_rows="
							<< tile.capacity_forced_dense_rows
							<< " conservative_capacity_locked_rows="
							<< tile.conservative_capacity_locked_rows
							<< " max_local_merge_runs="
							<< tile.packed.max_local_merge_runs
							<< " max_local_merge_bank_packets="
							<< tile.packed.max_local_merge_bank_packets
							<< " max_shard_beats=" << max_shard_beats
							<< " pack_ms=" << pack_ms << '\n';
						if (std::getenv("ADAPT_DEBUG_ROUTES") != nullptr) {
							for (id_t local_row = 0;
									local_row < tile.row_end - tile.row_begin;
									++local_row) {
								const uint32_t route = tile.packed.route[local_row];
								if ((route & kRouteModeMask) != 3U
										&& (route & kScalableCapacityLock) == 0)
									continue;
								std::cout << "scalable_debug_route window=" << index
									<< " batch=" << batch
									<< " row=" << tile.row_begin + local_row
									<< " local_row=" << local_row
									<< " mode=" << (route & kRouteModeMask)
									<< " products="
									<< ((route & ~kScalableCapacityLock)
										>> kRouteModeBits)
									<< " capacity_lock="
									<< ((route & kScalableCapacityLock) != 0)
									<< " tasks="
									<< tile.packed.row_task_ptr[local_row + 1]
										- tile.packed.row_task_ptr[local_row]
									<< " shard_tasks=";
								std::array<uint64_t, kShards> shard_tasks{};
								std::array<uint64_t, kShards> shard_products{};
								for (id_t task = tile.packed.row_task_ptr[local_row];
										task < tile.packed.row_task_ptr[local_row + 1];
										++task) {
									const TaskWord512& word = tile.packed.tasks[task >> 2];
									const uint64_t low = word.lane[(task & 3U) * 2];
									const unsigned shard = (low >> 56) & 7U;
									shard_tasks[shard]++;
									shard_products[shard] += (low >> 32) & 0xffffffU;
								}
								for (unsigned shard = 0; shard < kShards; ++shard)
									std::cout << (shard ? "/" : "")
										<< shard_tasks[shard];
								std::cout << " shard_products=";
								for (unsigned shard = 0; shard < kShards; ++shard)
									std::cout << (shard ? "/" : "")
										<< shard_products[shard];
								std::cout << '\n';
							}
						}
					}
				}
				std::cout << "SCALABLE_WINDOW_DRY_RUN: PASS\n";
				return 0;
			}
			const ScalableExecutionResult result = execute_scalable_windows(matrix,
				scalable_index, xclbin_path, argv[2], argv[4], reps);
			std::cout << "SCALABLE_WINDOW_BOARD: PASS execution_regions="
				<< result.tiles
				<< " kernel_ms=" << result.kernel_ms
				<< " packing_ms=" << result.packing_ms
				<< " h2d_ms=" << result.h2d_ms
				<< " d2h_ms=" << result.d2h_ms
				<< " verification_ms=" << result.verification_ms
				<< " output_nnz=" << result.output_nnz
				<< " route(merge,dense)=(" << result.merge_rows << ','
				<< result.dense_rows << ") default_route(merge,dense)=("
				<< result.default_merge_rows << ',' << result.default_dense_rows
				<< ") default_route_products(merge,dense)=("
				<< result.default_merge_products << ','
				<< result.default_dense_products
				<< ") structural_crosscheck_rows="
				<< result.structural_crosscheck_rows
				<< " baseline_crosscheck_rows="
				<< result.baseline_crosscheck_rows << " sampled_rows="
				<< result.sampled_rows << '\n';
			return 0;
		}
		const auto oracle_begin = Clock::now();
		const Reference ref = square_reference(matrix);
		const double cpu_oracle_ms = std::chrono::duration<double, std::milli>(
			Clock::now() - oracle_begin).count();
		const auto preflight_begin = Clock::now();
		uint64_t max_input_runs = 0;
		uint64_t max_input_products = 0;
		for (id_t row = 0; row < matrix.rows; ++row) {
			const id_t runs = matrix.rowptr[row + 1] - matrix.rowptr[row];
			uint64_t products = 0;
			for (id_t at = matrix.rowptr[row]; at < matrix.rowptr[row + 1]; ++at) {
				const id_t k = matrix.col[at];
				products += matrix.rowptr[k + 1] - matrix.rowptr[k];
			}
			max_input_runs = std::max<uint64_t>(max_input_runs, runs);
			max_input_products = std::max(max_input_products, products);
		}
		const double preflight_ms = std::chrono::duration<double, std::milli>(
			Clock::now() - preflight_begin).count();
		const auto selector_begin = Clock::now();
		adaptive_host::Plan plan = choose_routes(matrix);
		const double selector_ms = std::chrono::duration<double, std::milli>(
			Clock::now() - selector_begin).count();
		double kernel_mhz = 160.0;
		if (const char* value = std::getenv("ADAPT_KERNEL_MHZ")) {
			const double parsed = std::strtod(value, nullptr);
			if (std::isfinite(parsed) && parsed >= 50.0 && parsed <= 500.0)
				kernel_mhz = parsed;
		}
		// The lightweight policy intentionally uses thresholds rather than a
		// predicted cycle gain, so the legacy cycle-based amortization guard does
		// not apply.  Its own small-work and material-work guards are evaluated in
		// choose_light_plan().
		if (!g_used_light_selector)
			adaptive_host::account_for_host_overhead(plan, selector_ms, reps,
				0.03, kernel_mhz);
		const auto packing_begin = Clock::now();
		const PackedInput packed = pack_input(matrix, plan);
		// The current B-row-cache readers access external HBM directly.  The
		// former 12,288-beat whole-B limit belonged to an older on-chip cache and
		// must not reject otherwise legal streamed shards.
			uint64_t packed_route_products = 0;
			for (id_t route_word : packed.route)
				packed_route_products += route_word >> kRouteModeBits;
			if (packed_route_products != ref.products)
				throw std::runtime_error("route product metadata mismatch: packed="
					+ std::to_string(packed_route_products) + " reference="
					+ std::to_string(ref.products));
			const DmsaBatchSummary dmsa = summarize_dmsa_batches(packed, matrix.rows);
			const bool debug_dense_cache
				= std::getenv("ADAPT_DEBUG_DENSE_CACHE") != nullptr;
			const DenseReaderCacheSummary dense_cache = debug_dense_cache
				? summarize_dense_reader_cache(packed, matrix.rows, matrix.cols)
				: DenseReaderCacheSummary{};
			const double packing_ms = std::chrono::duration<double, std::milli>(
				Clock::now() - packing_begin).count();
		std::cout << "matrix=" << matrix.rows << 'x' << matrix.cols
			<< " nnz=" << matrix.col.size() << " products=" << ref.products
			<< " output_nnz=" << ref.col.size() << " tasks="
			<< packed.row_task_ptr.back()
			<< " matrix_load_ms=" << matrix_load_ms
			<< " cpu_oracle_ms=" << cpu_oracle_ms
			<< " preflight_ms=" << preflight_ms
			<< " packing_ms=" << packing_ms
			<< " max_input_runs=" << max_input_runs
			<< " max_input_products=" << max_input_products
			<< " max_local_merge_runs=" << packed.max_local_merge_runs
			<< " max_local_merge_bank_packets="
			<< packed.max_local_merge_bank_packets << '\n'
			<< "route(merge,dense)=(" << plan.merge_rows << ',' << plan.dense_rows
			<< ") model_mode=" << global_mode_name(plan.model_mode)
			<< " selected_mode=" << global_mode_name(plan.mode);
		if (plan.summary_work != 0) {
			std::cout << " predicted_cycles(merge,dense,adaptive)=("
				<< static_cast<uint64_t>(plan.merge_only_cycles) << ','
				<< static_cast<uint64_t>(plan.dense_only_cycles) << ','
				<< static_cast<uint64_t>(plan.adaptive_cycles) << ')'
				<< " dense_pairs(only,adaptive)=("
				<< plan.dense_only_pairs << ','
				<< plan.adaptive_dense_pairs << ')'
				<< " dense_quads(only,adaptive)=("
				<< plan.dense_only_quads << ','
				<< plan.adaptive_dense_quads << ')'
				<< " dense_octs(only,adaptive)=("
				<< plan.dense_only_octs << ','
				<< plan.adaptive_dense_octs << ')'
				<< " dense_hexes(only,adaptive)=("
				<< plan.dense_only_hexes << ','
				<< plan.adaptive_dense_hexes << ')'
				<< " dense_cycle_components(product,route,extract,setup,credit)=("
				<< static_cast<uint64_t>(plan.dense_only_product_cycles) << ','
				<< static_cast<uint64_t>(plan.dense_only_route_cycles) << ','
				<< static_cast<uint64_t>(plan.dense_only_extract_cycles) << ','
				<< static_cast<uint64_t>(plan.dense_only_setup_cycles) << ','
				<< static_cast<uint64_t>(plan.dense_only_batch_credit_cycles) << ')';
		} else {
			std::cout << " detailed_prediction=skipped";
		}
		std::cout << " route_products(merge,dense)=(" << plan.merge_products << ','
			<< plan.dense_products << ") profile(multi_rows,multi_products,"
			<< "wide_rows,wide_products)=(" << plan.multi_rows << ','
			<< plan.multi_products << ',' << plan.wide_rows << ','
			<< plan.wide_products << ") product_cv=" << plan.active_product_cv
			<< " selector_ms=" << selector_ms;
		if (g_used_light_selector) {
			std::cout << " light_selector(edges,hll_entries,eligible,ambiguous,"
				"sampled_rows,sampled_products,sampled_cr,dense_work_fraction)=("
				<< g_light_selector_report.cheap_edges << ','
				<< g_light_selector_report.hll_build_entries << ','
				<< g_light_selector_report.eligible_rows << ','
				<< g_light_selector_report.ambiguous_rows << ','
				<< g_light_selector_report.sampled_rows << ','
				<< g_light_selector_report.sampled_products << ','
				<< g_light_selector_report.sampled_compression << ','
				<< g_light_selector_report.dense_work_fraction << ')';
		}
		std::cout << " shard_work=";
		for (unsigned shard = 0; shard < kShards; ++shard)
			std::cout << (shard ? "," : "") << packed.shard_work[shard];
		std::cout << " shard_beats=";
		for (unsigned shard = 0; shard < kShards; ++shard)
			std::cout << (shard ? "," : "") << packed.B[shard].size();
		std::cout << " dmsa_merge_route_candidate(rows,products,batches,"
			"fallback_rows)=("
			<< dmsa.selected_rows << ',' << dmsa.selected_products << ','
			<< dmsa.batches << ',' << dmsa.fallback_rows << ')';
		if (debug_dense_cache) {
			const double command_hit_rate = dense_cache.cacheable_commands == 0
				? 0.0 : static_cast<double>(dense_cache.cache_hits)
					/ dense_cache.cacheable_commands;
			const double beat_save_rate = dense_cache.dense_beats == 0
				? 0.0 : static_cast<double>(dense_cache.saved_hbm_beats)
					/ dense_cache.dense_beats;
			std::cout << " dense_reader_cache(commands,cacheable,hits,"
				"dense_beats,saved_beats,command_hit_rate,beat_save_rate)=("
				<< dense_cache.dense_commands << ','
				<< dense_cache.cacheable_commands << ','
				<< dense_cache.cache_hits << ',' << dense_cache.dense_beats
				<< ',' << dense_cache.saved_hbm_beats << ','
				<< command_hit_rate << ',' << beat_save_rate << ')'
				<< " reader_commands=";
			for (unsigned shard = 0; shard < kShards; ++shard)
				std::cout << (shard ? "," : "")
					<< dense_cache.reader_commands[shard];
			std::cout << " reader_hits=";
			for (unsigned shard = 0; shard < kShards; ++shard)
				std::cout << (shard ? "," : "")
					<< dense_cache.reader_hits[shard];
		}
		std::cout << '\n';
		if (const char* debug_routes = std::getenv("ADAPT_DEBUG_ROUTES")) {
			id_t requested = 64;
			if (lower(debug_routes) == "all") requested = matrix.rows;
			else {
				char* end = nullptr;
				const unsigned long parsed = std::strtoul(debug_routes, &end, 10);
				if (end != debug_routes && *end == '\0' && parsed != 0)
					requested = static_cast<id_t>(std::min<unsigned long>(
						parsed, std::numeric_limits<id_t>::max()));
			}
			const id_t limit = std::min<id_t>(matrix.rows, requested);
			for (id_t row = 0; row < limit; ++row) {
				std::cout << "debug_route row=" << row
					<< " mode=" << (packed.route[row] & kRouteModeMask)
					<< " products=" << (packed.route[row] >> kRouteModeBits)
					<< " tasks=" << packed.row_task_ptr[row + 1]
						- packed.row_task_ptr[row] << '\n';
			}
		}
		if (dry_run) {
			std::cout << "DRY_RUN: task packing, sharding and CPU oracle PASS\n";
			return 0;
		}

		const auto runtime_setup_begin = Clock::now();
		if (lower(argv[2]) == "auto" && std::getenv("TARGET_CARD") == nullptr)
			::setenv("TARGET_CARD", "u280", 0);
		xrt::device device(host_device::parse_device_arg(argv[2]));
		const auto uuid = device.load_xclbin(xclbin_path);
		xrt::kernel kernel(device, uuid, "adaptive_hbm_spgemm:{adaptive_hbm_spgemm_1}",
			xrt::kernel::cu_access_mode::exclusive);
		xrt::bo task_bo(device, packed.tasks.size() * sizeof(TaskWord512), kernel.group_id(0));
		xrt::bo row_bo(device, packed.row_task_ptr.size() * sizeof(id_t), kernel.group_id(1));
		xrt::bo route_bo(device, packed.route.size() * sizeof(id_t), kernel.group_id(2));
		std::vector<xrt::bo> B_bo;
		for (unsigned shard = 0; shard < kShards; ++shard)
			B_bo.emplace_back(device, packed.B[shard].size() * sizeof(Beat512),
				kernel.group_id(3 + shard));
		xrt::bo C_row_bo(device, (matrix.rows + 1) * sizeof(id_t), kernel.group_id(23));
		const size_t C_words = (ref.col.size() + 7) / 8;
		const size_t C_words_per_bank = std::max<size_t>(1, (C_words + 3) / 4);
		std::array<xrt::bo, 4> C_item_bo = {
			xrt::bo(device, C_words_per_bank * sizeof(Beat512), kernel.group_id(24)),
			xrt::bo(device, C_words_per_bank * sizeof(Beat512), kernel.group_id(25)),
			xrt::bo(device, C_words_per_bank * sizeof(Beat512), kernel.group_id(26)),
			xrt::bo(device, C_words_per_bank * sizeof(Beat512), kernel.group_id(27))};
		xrt::bo stats_bo(device, kStats * sizeof(id_t), kernel.group_id(28));
		const double runtime_setup_ms = std::chrono::duration<double, std::milli>(
			Clock::now() - runtime_setup_begin).count();
		const auto static_stage_begin = Clock::now();
		std::memcpy(task_bo.map<void*>(), packed.tasks.data(),
			packed.tasks.size() * sizeof(TaskWord512));
		std::memcpy(row_bo.map<void*>(), packed.row_task_ptr.data(),
			packed.row_task_ptr.size() * sizeof(id_t));
		for (unsigned shard = 0; shard < kShards; ++shard)
			std::memcpy(B_bo[shard].map<void*>(), packed.B[shard].data(),
				packed.B[shard].size() * sizeof(Beat512));
		const double static_stage_ms = std::chrono::duration<double, std::milli>(
			Clock::now() - static_stage_begin).count();
		const auto route_stage_begin = Clock::now();
		std::memcpy(route_bo.map<void*>(), packed.route.data(),
			packed.route.size() * sizeof(id_t));
		std::memset(stats_bo.map<void*>(), 0, kStats * sizeof(id_t));
		const double route_stage_ms = std::chrono::duration<double, std::milli>(
			Clock::now() - route_stage_begin).count();
		const auto static_h2d_begin = Clock::now();
		task_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
		row_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
		for (auto& bo : B_bo) bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
		const double static_h2d_ms = std::chrono::duration<double, std::milli>(
			Clock::now() - static_h2d_begin).count();
		const auto route_h2d_begin = Clock::now();
		route_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
		stats_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
		const double route_h2d_ms = std::chrono::duration<double, std::milli>(
			Clock::now() - route_h2d_begin).count();
		std::cout << "runtime_setup_ms=" << runtime_setup_ms
			<< " static_stage_ms=" << static_stage_ms
			<< " route_stage_ms=" << route_stage_ms
			<< " static_h2d_ms=" << static_h2d_ms
			<< " route_h2d_ms=" << route_h2d_ms << '\n';

		// A cheap MERGE_ONLY plan already carries EMPTY/DIRECT/MERGE tags and no
		// dense row.  Preserve those tags so structurally empty products are not
		// needlessly materialized by the force-merge diagnostic override.
		const ForceMode selected_force = plan.mode == adaptive_host::MERGE_ONLY
				&& plan.dense_rows == 0 ? kUseRoute : force_mode_for(plan.mode);
		std::vector<std::pair<std::string, ForceMode>> modes;
		modes.push_back({std::string("host_selected_")
			+ global_mode_name(plan.mode), selected_force});
		const std::vector<std::pair<std::string, ForceMode>> comparisons = {
			{"row_adaptive", kUseRoute},
			{"merge_preferred_capacity_safe", kForceMerge},
			{"dense_only", kForceDense}};
		for (const auto& candidate : comparisons)
			if (candidate.second != selected_force) modes.push_back(candidate);
		// Optional board-debug filter.  The default campaign still exercises the
		// host-selected mode plus every distinct oracle mode.  Selecting one mode
		// here makes it possible to isolate a MERGE, DENSE, or packed-output fault
		// without rebuilding the XCLBIN or waiting for earlier modes to verify.
		if (const char* requested_mode = std::getenv("ADAPT_TEST_MODE")) {
			const std::string requested = lower(requested_mode);
			if (requested == "row_adaptive" || requested == "adaptive")
				modes = {{"row_adaptive", kUseRoute}};
			else if (requested == "merge_only" || requested == "merge"
					|| requested == "merge_preferred_capacity_safe")
				modes = {{"merge_preferred_capacity_safe", kForceMerge}};
			else if (requested == "dense_only" || requested == "dense")
				modes = {{"dense_only", kForceDense}};
			else
				throw std::runtime_error("invalid ADAPT_TEST_MODE: "
					+ requested);
		}
		std::vector<ModeResult> results;
		for (const auto& mode : modes) {
			// Prime the command path, HBM pages and kernel state once.  This run is
			// deliberately excluded from both the reported median and system total.
			auto warmup = kernel(task_bo, row_bo, route_bo,
				B_bo[0], B_bo[1], B_bo[2], B_bo[3],
				B_bo[4], B_bo[5], B_bo[6], B_bo[7],
				static_cast<id_t>(packed.B[0].size()),
				static_cast<id_t>(packed.B[1].size()),
				static_cast<id_t>(packed.B[2].size()),
				static_cast<id_t>(packed.B[3].size()),
				static_cast<id_t>(packed.B[4].size()),
				static_cast<id_t>(packed.B[5].size()),
				static_cast<id_t>(packed.B[6].size()),
				static_cast<id_t>(packed.B[7].size()),
				matrix.rows, matrix.cols, static_cast<uint32_t>(mode.second),
				static_cast<id_t>(ref.col.size()), C_row_bo,
				C_item_bo[0], C_item_bo[1], C_item_bo[2], C_item_bo[3], stats_bo);
			warmup.wait();
			std::vector<double> times;
			std::vector<double> d2h_times;
			std::vector<id_t> rowptr(matrix.rows + 1);
			std::vector<uint64_t> item(std::max<size_t>(1, ref.col.size()));
			std::vector<id_t> stat(kStats);
			for (unsigned rep = 0; rep < reps; ++rep) {
				const auto begin = Clock::now();
				auto run = kernel(task_bo, row_bo, route_bo,
					B_bo[0], B_bo[1], B_bo[2], B_bo[3],
					B_bo[4], B_bo[5], B_bo[6], B_bo[7],
					static_cast<id_t>(packed.B[0].size()),
					static_cast<id_t>(packed.B[1].size()),
					static_cast<id_t>(packed.B[2].size()),
					static_cast<id_t>(packed.B[3].size()),
					static_cast<id_t>(packed.B[4].size()),
					static_cast<id_t>(packed.B[5].size()),
					static_cast<id_t>(packed.B[6].size()),
					static_cast<id_t>(packed.B[7].size()),
					matrix.rows, matrix.cols, static_cast<uint32_t>(mode.second),
					static_cast<id_t>(ref.col.size()), C_row_bo,
					C_item_bo[0], C_item_bo[1], C_item_bo[2], C_item_bo[3], stats_bo);
				run.wait();
				times.push_back(std::chrono::duration<double, std::milli>(
					Clock::now() - begin).count());
				const auto d2h_begin = Clock::now();
				C_row_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
				for (auto& bank : C_item_bo)
					bank.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
				stats_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
				std::memcpy(rowptr.data(), C_row_bo.map<void*>(),
					rowptr.size() * sizeof(id_t));
				const std::array<const Beat512*, 4> C_bank_map = {
					C_item_bo[0].map<const Beat512*>(),
					C_item_bo[1].map<const Beat512*>(),
					C_item_bo[2].map<const Beat512*>(),
					C_item_bo[3].map<const Beat512*>()};
				for (size_t index = 0; index < item.size(); ++index) {
					const size_t word = index >> 3;
					item[index] = C_bank_map[word & 3][word >> 2].lane[index & 7];
				}
				std::memcpy(stat.data(), stats_bo.map<void*>(),
					stat.size() * sizeof(id_t));
				d2h_times.push_back(std::chrono::duration<double, std::milli>(
					Clock::now() - d2h_begin).count());
			}
			const double d2h_ms = median(d2h_times);
			const double kernel_ms = median(times);
			// Preserve every measured sample in the captured board log.  The CSV
			// keeps the standard median for compact tables, while these samples
			// make the statistic independently reproducible and expose run-to-run
			// dispersion without including the warm-up execution.
			std::cout << "timing_samples mode=" << mode.first
				<< " kernel_ms=[";
			for (size_t index = 0; index < times.size(); ++index) {
				if (index != 0) std::cout << ',';
				std::cout << std::setprecision(9) << times[index];
			}
			std::cout << "] d2h_ms=[";
			for (size_t index = 0; index < d2h_times.size(); ++index) {
				if (index != 0) std::cout << ',';
				std::cout << std::setprecision(9) << d2h_times[index];
			}
			std::cout << "]\n";
			if (std::getenv("ADAPT_DIAGNOSTIC_PREVERIFY") != nullptr) {
				std::cout << "preverify mode=" << mode.first
					<< " kernel_ms=" << kernel_ms
					<< " output_nnz=" << stat[9]
					<< " replay=" << stat[7]
					<< " raw=" << stat[8]
					<< " route_beats=" << stat[12]
					<< " merge_stream=" << stat[kStatMergeStreamCycles]
					<< '\n';
				if (stat[kStatDiagnosticVersion] == kDenseDiagnosticVersion) {
					std::cout << "preverify_dense_stage mode=" << mode.first
						<< " collect=" << stat[kStatDenseCollectCycles]
						<< " epoch_clear=" << stat[kStatDenseEpochClearCycles]
						<< " bank_products=[";
					for (unsigned bank = 0; bank < kShards; ++bank) {
						if (bank != 0) std::cout << ',';
						std::cout << stat[kStatDenseBankProducts + bank];
					}
					std::cout << "] bank_unique_addresses=[";
					for (unsigned bank = 0; bank < kShards; ++bank) {
						if (bank != 0) std::cout << ',';
						std::cout << stat[kStatDenseBankUniqueAddresses + bank];
					}
					std::cout << "] touched=" << stat[kStatDenseTouchedAddresses]
						<< " word_scans=" << stat[kStatDenseBitmapWordScans]
						<< " nonempty_words=" << stat[kStatDenseNonemptyWords]
						<< " active_groups=" << stat[kStatDenseActiveAddressGroups]
						<< " candidates=" << stat[kStatDenseCandidateColumns]
						<< " extract=" << stat[kStatDenseExtractIterations]
						<< " output_packets=" << stat[kStatDenseOutputPackets]
						<< " coalesced_products="
						<< stat[kStatDenseCoalescedProducts] << '\n';
				}
			}
			// Diagnostic only: determine whether a mismatch is confined to the
			// output-count statistic or whether the CSR payload itself is wrong.
			if (std::getenv("ADAPT_DEBUG_CLAMP_STATS") != nullptr
					&& !rowptr.empty() && rowptr.back() <= item.size()) {
				std::cout << "debug_clamp_stats_output=" << stat[9]
					<< "->" << rowptr.back() << '\n';
				stat[9] = rowptr.back();
			}
			const auto verification_begin = Clock::now();
			verify(ref, rowptr, item, stat);
			const double verification_ms = std::chrono::duration<double, std::milli>(
				Clock::now() - verification_begin).count();
			results.push_back({mode.first, kernel_ms, d2h_ms,
				verification_ms, stat});
			std::cout << mode.first << " kernel_ms=" << results.back().kernel_ms
				<< " d2h_ms=" << d2h_ms
				<< " verification_ms=" << verification_ms
				<< " merge_rows=" << stat[3] << " dense_rows=" << stat[4]
				<< " replay=" << stat[7] << " raw=" << stat[8]
				<< " route_beats=" << stat[12] << " PASS\n";
			if (stat[kStatDiagnosticVersion] == kDenseDiagnosticVersion) {
				std::cout << "dense_stage_diag mode=" << mode.first
					<< " collect=" << stat[kStatDenseCollectCycles]
					<< " epoch_clear=" << stat[kStatDenseEpochClearCycles]
					<< " bank_products=[";
				for (unsigned bank = 0; bank < kShards; ++bank) {
					if (bank != 0) std::cout << ',';
					std::cout << stat[kStatDenseBankProducts + bank];
				}
				std::cout << "] bank_unique_addresses=[";
				for (unsigned bank = 0; bank < kShards; ++bank) {
					if (bank != 0) std::cout << ',';
					std::cout << stat[kStatDenseBankUniqueAddresses + bank];
				}
				std::cout << "] touched=" << stat[kStatDenseTouchedAddresses]
					<< " bitmap_clear=" << stat[kStatDenseBitmapClearCycles]
					<< " bitmap_build=" << stat[kStatDenseBitmapBuildCycles]
					<< " word_scans=" << stat[kStatDenseBitmapWordScans]
					<< " nonempty_words=" << stat[kStatDenseNonemptyWords]
					<< " active_groups=" << stat[kStatDenseActiveAddressGroups]
					<< " candidates=" << stat[kStatDenseCandidateColumns]
					<< " extract=" << stat[kStatDenseExtractIterations]
					<< " output_packets=" << stat[kStatDenseOutputPackets]
					<< " merge_stream=" << stat[kStatMergeStreamCycles]
					<< " coalesced_products="
					<< stat[kStatDenseCoalescedProducts]
					<< '\n';
			}
		}
		std::ofstream output(argv[4]);
		output << "matrix,mode,warmup_runs,reps,kernel_ms,d2h_ms,verification_ms,"
			"matrix_load_ms,cpu_oracle_ms,preflight_ms,selector_ms,packing_ms,"
			"runtime_setup_ms,static_stage_ms,route_stage_ms,static_h2d_ms,"
			"route_h2d_ms,selected_system_one_time_ms,"
			"selected_system_one_shot_ms,selected_system_batch_total_ms,"
			"selected_system_amortized_ms,rows,input_nnz,products,output_nnz,"
			"merge_rows,dense_rows,replay,raw,route_beats,merge_stream_cycles,"
			"dense_coalesced_products,verified\n";
		const double system_one_time_ms = selector_ms + packing_ms
			+ static_stage_ms + route_stage_ms + static_h2d_ms + route_h2d_ms;
		const double selected_system_one_shot_ms = system_one_time_ms
			+ results.front().kernel_ms + results.front().d2h_ms;
		const double selected_system_batch_total_ms = system_one_time_ms
			+ reps * (results.front().kernel_ms + results.front().d2h_ms);
		const double selected_system_amortized_ms =
			selected_system_batch_total_ms / reps;
		for (size_t result_index = 0; result_index < results.size(); ++result_index) {
			const auto& result = results[result_index];
			output << argv[3] << ',' << result.mode << ",1," << std::fixed
				<< reps << ',' << std::setprecision(6) << result.kernel_ms << ','
				<< result.d2h_ms << ',' << result.verification_ms << ','
				<< matrix_load_ms << ',' << cpu_oracle_ms << ',' << preflight_ms
				<< ',' << selector_ms << ',' << packing_ms << ',' << runtime_setup_ms
				<< ',' << static_stage_ms << ',' << route_stage_ms << ','
				<< static_h2d_ms << ',' << route_h2d_ms << ',';
			if (result_index == 0)
				output << system_one_time_ms << ','
					<< selected_system_one_shot_ms << ','
					<< selected_system_batch_total_ms << ','
					<< selected_system_amortized_ms;
			else
				output << ",,,";
			output << ',' << matrix.rows
				<< ',' << matrix.col.size() << ',' << ref.products << ','
				<< ref.col.size() << ',' << result.stats[3] << ',' << result.stats[4]
				<< ',' << result.stats[7] << ',' << result.stats[8] << ','
				<< result.stats[12] << ',' << result.stats[kStatMergeStreamCycles]
				<< ',' << result.stats[kStatDenseCoalescedProducts] << ",1\n";
		}
		const auto oracle = std::min_element(results.begin(), results.end(),
			[](const ModeResult& left, const ModeResult& right) {
				return left.kernel_ms < right.kernel_ms;
			});
		const double regret = 100.0 * (results.front().kernel_ms
			- oracle->kernel_ms) / oracle->kernel_ms;
		std::cout << "host_decision=" << global_mode_name(plan.mode)
			<< " reason=" << plan.decision_reason
			<< " oracle=" << oracle->mode
			<< " host_regret_pct=" << regret
			<< " selected_system_one_time_ms=" << system_one_time_ms
			<< " selected_system_one_shot_ms=" << selected_system_one_shot_ms
			<< " selected_system_batch_total_ms="
			<< selected_system_batch_total_ms
			<< " selected_system_amortized_ms="
			<< selected_system_amortized_ms << '\n';
		std::cout << "wrote " << argv[4] << '\n';
		return 0;
	} catch (const std::exception& error) {
		std::cerr << "ERROR: " << error.what() << '\n';
		return 1;
	}
}
