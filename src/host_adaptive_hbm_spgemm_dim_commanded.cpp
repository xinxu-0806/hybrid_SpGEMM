// Leda-style dimension-command Host built on the deployed scalable packer.
// Keep the proven legacy Host translation unit intact and reuse its Matrix
// Market loader, selector, capacity splitter, verifier, and wire ABI here.
#define ADAPT_RUNTIME_WINDOW_COLUMNS 1
#define main adaptive_scalable_legacy_main
#include "host_adaptive_hbm_spgemm_scalable.cpp"
#undef main
#undef ADAPT_RUNTIME_WINDOW_COLUMNS
#include "rowpersistent_selector_policy_robust_v3_generated.h"

namespace {

constexpr uint64_t kDimTargetTasks = 15000000;
constexpr uint64_t kDimTargetBankBytes = 240ULL << 20;
constexpr uint32_t kDimCommandMagic = 0x444d4331U;
constexpr uint32_t kRowPersistentCommandMagic = 0x52504331U;

// Row-persistent mode performs exactly one selector pass over complete logical
// rows.  A 64K physical DENSE workspace is then reused for every disjoint
// output-column fragment.  Fragmentation is only a storage mechanism: all
// fragments inherit the complete row's route and are emitted in row-major
// order so that only the final fragment closes the logical CSR row.
bool g_dim_row_persistent_whole_row = false;
adaptive_host::GlobalMode g_dim_host_selected_mode = adaptive_host::MERGE_ONLY;

struct DimTileMeta {
	uint32_t window = 0;
	uint32_t width = 0;
	id_t row_begin = 0;
	id_t row_end = 0;
	id_t output_row_base = 0;
	id_t virtual_row_base = 0;
	uint64_t products = 0;
	uint64_t output_upper = 0;
	adaptive_host::Plan plan;
	uint64_t capacity_forced_dense_rows = 0;
	uint64_t conservative_capacity_locked_rows = 0;
};

struct DimSuperbatch {
	std::vector<Beat512> commands;
	PackedInput packed;
	std::vector<DimTileMeta> tiles;
	id_t task_count = 0;
	id_t physical_rows = 0;
	id_t total_rows = 0;
	id_t virtual_row_extent = 0;
	uint64_t products = 0;
	uint64_t output_upper = 0;
	uint64_t default_merge_rows = 0;
	uint64_t default_dense_rows = 0;
	uint64_t default_merge_products = 0;
	uint64_t default_dense_products = 0;
	double packing_ms = 0.0;
	bool row_persistent = false;
	std::vector<id_t> logical_rows;
};

struct DimWindowPayload {
	id_t task_base = 0;
	id_t rowptr_base = 0;
	id_t route_base = 0;
	uint32_t window = 0;
	uint32_t width = 0;
	id_t row_begin = 0;
	id_t row_end = 0;
	std::vector<id_t> row_task_ptr;
	std::vector<id_t> route;
	uint64_t products = 0;
	uint64_t output_upper = 0;
};

struct DimModeTotal {
	std::string name;
	ForceMode force = kUseRoute;
	std::vector<double> samples;
	double h2d_ms = 0.0;
	double d2h_ms = 0.0;
	double verification_ms = 0.0;
	uint64_t output_nnz = 0;
	uint64_t merge_rows = 0;
	uint64_t dense_rows = 0;
	uint64_t sampled_rows = 0;
	uint64_t commands = 0;
	uint64_t rows = 0;
	uint64_t products = 0;
	uint64_t default_merge_rows = 0;
	uint64_t default_dense_rows = 0;
	uint64_t default_merge_products = 0;
	uint64_t default_dense_products = 0;
	double packing_ms = 0.0;
};

static uint64_t dim_output_bank_bytes(uint64_t items) {
	const uint64_t words = (items + 7) / 8;
	const uint64_t words_per_bank = (words + 3) / 4;
	return words_per_bank * sizeof(Beat512);
}

static bool dim_can_append(const DimSuperbatch& batch,
		const WindowPackResult& tile) {
	const id_t tile_tasks = tile.packed.row_task_ptr.back();
	const uint64_t rows = tile.row_end - tile.row_begin;
	if (batch.commands.size() >= 256
			|| uint64_t(batch.task_count) + tile_tasks > kDimTargetTasks
			|| uint64_t(batch.total_rows) + rows
				> std::numeric_limits<id_t>::max())
		return false;
	if (dim_output_bank_bytes(
			batch.output_upper + tile.output_nnz_upper_bound)
			> kDimTargetBankBytes)
		return false;
	const uint64_t task_words =
		(uint64_t(batch.task_count) + tile_tasks + 3) / 4;
	if (task_words * sizeof(TaskWord512) > kDimTargetBankBytes)
		return false;
	const uint64_t row_bytes =
		(uint64_t(batch.packed.row_task_ptr.size())
			+ tile.packed.row_task_ptr.size()) * sizeof(id_t);
	const uint64_t route_bytes =
		(uint64_t(batch.packed.route.size()) + tile.packed.route.size())
		* sizeof(id_t);
	if (row_bytes + route_bytes + kStats * sizeof(id_t)
			> kDimTargetBankBytes)
		return false;
	for (unsigned shard = 0; shard < kShards; ++shard)
		if ((uint64_t(batch.packed.B[shard].size())
				+ tile.packed.B[shard].size()) * sizeof(Beat512)
				> kDimTargetBankBytes)
			return false;
	return true;
}

static void dim_store_task(PackedInput* packed, id_t task,
		uint64_t low, uint64_t high) {
	const size_t word = task >> 2;
	if (packed->tasks.size() <= word) packed->tasks.resize(word + 1);
	const unsigned lane = (task & 3U) * 2U;
	packed->tasks[word].lane[lane] = low;
	packed->tasks[word].lane[lane + 1] = high;
}

static Beat512 dim_make_command(id_t task_base, id_t rowptr_base,
		id_t route_base, id_t rows, id_t columns, id_t virtual_row_base,
		uint32_t window, id_t global_row_begin, id_t dense_base = 0,
		bool logical_row_last = true) {
	Beat512 command{};
	command.lane[0] = uint64_t(task_base) | (uint64_t(rowptr_base) << 32);
	command.lane[1] = uint64_t(route_base) | (uint64_t(rows) << 32);
	command.lane[2] = uint64_t(columns) | (uint64_t(virtual_row_base) << 32);
	if (g_dim_row_persistent_whole_row) {
		// RPC1 lane 6 carries the global base of the physical DENSE workspace;
		// bit 224 (upper half of lane 3) alone closes the logical CSR row.
		command.lane[3] = uint64_t(dense_base)
			| (uint64_t(logical_row_last) << 32);
		command.lane[7] = uint64_t(kRowPersistentCommandMagic) << 32;
	} else {
		command.lane[3]
			= uint64_t(window) | (uint64_t(global_row_begin) << 32);
		command.lane[7] = uint64_t(kDimCommandMagic) << 32;
	}
	return command;
}

static adaptive_host::Plan dim_choose_whole_row_routes(const Csr& matrix) {
	adaptive_host::LightSelectorConfig config;
	config.row_local_dense = true;
	config.allow_segmented_dense = true;
	if (std::getenv("ADAPT_LIGHT_NO_SAMPLING") != nullptr)
		config.enable_sampling = false;
	adaptive_host::LightSelectorReport report;
	adaptive_host::Plan plan = adaptive_host::choose_light_plan(
		matrix.rowptr, matrix.col, matrix.cols, config, &report);

	// The row classifier exposes every plausible DENSE island.  The generated
	// matrix-level policy was fit only on the 38-matrix representative
	// validation set; three explicit exclusions are recorded and held-out test
	// results were unavailable.  It changes only the global execution mode:
	// the already computed per-row routes remain the ADAPTIVE implementation.
	const uint64_t routed_rows = static_cast<uint64_t>(plan.merge_rows)
		+ static_cast<uint64_t>(plan.dense_rows);
	const double dense_row_fraction = routed_rows == 0 ? 0.0
		: static_cast<double>(plan.dense_rows)
			/ static_cast<double>(routed_rows);
	const auto deployment_decision = rowpersistent_selector_policy::decide(
		report.dense_work_fraction, dense_row_fraction,
		plan.merge_rows, plan.dense_rows);
	const char* allow_uncalibrated =
		std::getenv("ADAPT_ENABLE_UNCALIBRATED_DENSE");
	const bool calibrated_merge_gate = allow_uncalibrated == nullptr
		|| std::strcmp(allow_uncalibrated, "0") == 0;
	if (calibrated_merge_gate) {
		if (deployment_decision == rowpersistent_selector_policy::kRowAdaptive) {
			plan.mode = adaptive_host::ADAPTIVE;
			plan.model_mode = adaptive_host::ADAPTIVE;
			plan.decision_reason
				= "rowpersistent-robust-v3-validation-adaptive";
		} else if (deployment_decision
				== rowpersistent_selector_policy::kDenseOnly) {
			plan.mode = adaptive_host::DENSE_ONLY;
			plan.model_mode = adaptive_host::DENSE_ONLY;
			plan.decision_reason
				= "rowpersistent-robust-v3-validation-dense-only";
		} else {
			plan.mode = adaptive_host::MERGE_ONLY;
			plan.model_mode = adaptive_host::MERGE_ONLY;
			plan.predicted_adaptive_gain_cycles = 0.0;
			plan.decision_reason
				= "rowpersistent-robust-v3-validation-merge-only";
		}
	}
	std::cout << "whole_row_selector mode=" << global_mode_name(plan.mode)
		<< " reason=" << plan.decision_reason
		<< " policy_sha=" << rowpersistent_selector_policy::kPolicySha256
		<< " route(merge,dense)=(" << plan.merge_rows << ','
		<< plan.dense_rows << ") sampled_rows=" << report.sampled_rows
		<< " sampled_products=" << report.sampled_products
		<< " ambiguous_rows=" << report.ambiguous_rows
		<< " hll_entries=" << report.hll_build_entries
		<< " dense_work_fraction=" << report.dense_work_fraction
		<< " dense_row_fraction=" << dense_row_fraction << '\n';
	return plan;
}

static void dim_apply_whole_row_routes(WindowPackResult* tile,
		const adaptive_host::Plan& whole_plan) {
	tile->plan = adaptive_host::Plan{};
	tile->plan.route.resize(tile->row_end - tile->row_begin, 0);
	for (id_t local_row = 0; local_row < tile->row_end - tile->row_begin;
			++local_row) {
		const id_t task_begin = tile->packed.row_task_ptr[local_row];
		const id_t task_end = tile->packed.row_task_ptr[local_row + 1];
		const id_t task_count = task_end - task_begin;
		const id_t global_row = tile->row_begin + local_row;
		uint32_t selected = task_count == 0 ? 0 : (task_count == 1 ? 1
			: (whole_plan.route.at(global_row) == 3 ? 3U : 2U));
		const uint32_t old_word = tile->packed.route[local_row];
		if (selected == 2 && (old_word & kScalableCapacityLock) != 0)
			throw std::runtime_error("whole-row MERGE route for row "
				+ std::to_string(global_row)
				+ " exceeds the physical local-MERGE capacity; use the "
					"row-major capacity-fragment packer");
		tile->packed.route[local_row] = (old_word & ~kRouteModeMask) | selected;
		tile->plan.route[local_row] = selected;
		const uint64_t products = old_word >> kRouteModeBits;
		if (selected == 0) tile->plan.empty_rows++;
		else if (selected == 1) tile->plan.direct_rows++;
		else if (selected == 2) {
			tile->plan.merge_rows++;
			tile->plan.merge_products += products;
		} else {
			tile->plan.dense_rows++;
			tile->plan.dense_products += products;
		}
	}
	tile->plan.mode = tile->plan.dense_rows == 0
		? adaptive_host::MERGE_ONLY
		: (tile->plan.merge_rows == 0 ? adaptive_host::DENSE_ONLY
			: adaptive_host::ADAPTIVE);
	tile->plan.model_mode = tile->plan.mode;
}

static void dim_append_tile(DimSuperbatch* batch,
		const WindowPackResult& tile, uint32_t width, double packing_ms) {
	if (!dim_can_append(*batch, tile))
		throw std::runtime_error("internal dimension super-batch capacity error");
	const id_t task_base = batch->task_count;
	const id_t rowptr_base = batch->packed.row_task_ptr.size();
	const id_t route_base = batch->packed.route.size();
	const id_t rows = tile.row_end - tile.row_begin;
	const id_t virtual_base = (batch->virtual_row_extent + 1) & ~id_t(1);
	std::array<id_t, kShards> B_base{};
	for (unsigned shard = 0; shard < kShards; ++shard)
		B_base[shard] = batch->packed.B[shard].size();

	const id_t tile_tasks = tile.packed.row_task_ptr.back();
	for (id_t local_task = 0; local_task < tile_tasks; ++local_task) {
		const TaskWord512& source = tile.packed.tasks[local_task >> 2];
		const unsigned source_lane = (local_task & 3U) * 2U;
		uint64_t low = source.lane[source_lane];
		const uint64_t high = source.lane[source_lane + 1];
		const unsigned shard = (low >> 56) & 7U;
		const uint64_t old_offset = static_cast<uint32_t>(low);
		const uint64_t new_offset = old_offset + B_base[shard];
		if (new_offset > std::numeric_limits<id_t>::max())
			throw std::runtime_error("dimension command B offset exceeds uint32");
		low = (low & 0xffffffff00000000ULL) | new_offset;
		dim_store_task(&batch->packed, task_base + local_task, low, high);
	}
	batch->task_count += tile_tasks;
	batch->packed.row_task_ptr.insert(batch->packed.row_task_ptr.end(),
		tile.packed.row_task_ptr.begin(), tile.packed.row_task_ptr.end());
	batch->packed.route.insert(batch->packed.route.end(),
		tile.packed.route.begin(), tile.packed.route.end());
	for (unsigned shard = 0; shard < kShards; ++shard)
		batch->packed.B[shard].insert(batch->packed.B[shard].end(),
			tile.packed.B[shard].begin(), tile.packed.B[shard].end());

	batch->commands.push_back(dim_make_command(task_base, rowptr_base,
		route_base, rows, width, virtual_base, tile.window, tile.row_begin));
	batch->physical_rows += rows;
	if (g_dim_row_persistent_whole_row) {
		batch->row_persistent = true;
		for (id_t row = tile.row_begin; row < tile.row_end; ++row)
			batch->logical_rows.push_back(row);
	}
	DimTileMeta meta;
	meta.window = tile.window;
	meta.width = width;
	meta.row_begin = tile.row_begin;
	meta.row_end = tile.row_end;
	meta.output_row_base = batch->total_rows;
	meta.virtual_row_base = virtual_base;
	meta.products = tile.products;
	meta.output_upper = tile.output_nnz_upper_bound;
	meta.plan = tile.plan;
	meta.capacity_forced_dense_rows = tile.capacity_forced_dense_rows;
	meta.conservative_capacity_locked_rows
		= tile.conservative_capacity_locked_rows;
	batch->tiles.push_back(std::move(meta));
	batch->total_rows += rows;
	batch->virtual_row_extent = virtual_base + rows;
	batch->products += tile.products;
	batch->output_upper += tile.output_nnz_upper_bound;
	batch->default_merge_rows += tile.plan.merge_rows;
	batch->default_dense_rows += tile.plan.dense_rows;
	batch->default_merge_products += tile.plan.merge_products;
	batch->default_dense_products += tile.plan.dense_products;
	batch->packing_ms += packing_ms;
}

static void dim_promote_window_columns_to_global(WindowPackResult* tile,
		id_t dense_base) {
	if (dense_base == 0) return;
	for (auto& shard : tile->packed.B)
		for (Beat512& beat : shard)
			for (uint64_t& lane : beat.lane) {
				const uint32_t local_col = static_cast<uint32_t>(lane);
				lane = (lane & 0xffffffff00000000ULL)
					| static_cast<uint32_t>(dense_base + local_col);
			}
}

static DimWindowPayload dim_append_window_payload(DimSuperbatch* batch,
		WindowPackResult tile, uint32_t width, double packing_ms) {
	DimWindowPayload payload;
	payload.task_base = batch->task_count;
	payload.rowptr_base = batch->packed.row_task_ptr.size();
	payload.route_base = batch->packed.route.size();
	payload.window = tile.window;
	payload.width = width;
	payload.row_begin = tile.row_begin;
	payload.row_end = tile.row_end;
	payload.row_task_ptr = tile.packed.row_task_ptr;
	payload.route = tile.packed.route;
	payload.products = tile.products;
	payload.output_upper = tile.output_nnz_upper_bound;

	const id_t dense_base = static_cast<id_t>(
		uint64_t(tile.window) * kWindowColumns);
	dim_promote_window_columns_to_global(&tile, dense_base);
	std::array<id_t, kShards> B_base{};
	for (unsigned shard = 0; shard < kShards; ++shard)
		B_base[shard] = batch->packed.B[shard].size();
	const id_t tile_tasks = tile.packed.row_task_ptr.back();
	for (id_t local_task = 0; local_task < tile_tasks; ++local_task) {
		const TaskWord512& source = tile.packed.tasks[local_task >> 2];
		const unsigned source_lane = (local_task & 3U) * 2U;
		uint64_t low = source.lane[source_lane];
		const uint64_t high = source.lane[source_lane + 1];
		const unsigned shard = (low >> 56) & 7U;
		const uint64_t new_offset = static_cast<uint32_t>(low) + B_base[shard];
		if (new_offset > std::numeric_limits<id_t>::max())
			throw std::runtime_error("row-persistent B offset exceeds uint32");
		low = (low & 0xffffffff00000000ULL) | new_offset;
		dim_store_task(&batch->packed, payload.task_base + local_task, low, high);
	}
	batch->task_count += tile_tasks;
	batch->packed.row_task_ptr.insert(batch->packed.row_task_ptr.end(),
		tile.packed.row_task_ptr.begin(), tile.packed.row_task_ptr.end());
	batch->packed.route.insert(batch->packed.route.end(),
		tile.packed.route.begin(), tile.packed.route.end());
	for (unsigned shard = 0; shard < kShards; ++shard)
		batch->packed.B[shard].insert(batch->packed.B[shard].end(),
			tile.packed.B[shard].begin(), tile.packed.B[shard].end());
	batch->products += tile.products;
	batch->output_upper += tile.output_nnz_upper_bound;
	batch->packing_ms += packing_ms;
	return payload;
}

static bool dim_superbatch_fits(const DimSuperbatch& batch) {
	if (uint64_t(batch.commands.size()) * sizeof(Beat512)
			> kDimTargetBankBytes
			|| batch.task_count > kDimTargetTasks
			|| batch.total_rows > std::numeric_limits<id_t>::max()
			|| dim_output_bank_bytes(batch.output_upper) > kDimTargetBankBytes)
		return false;
	if (batch.packed.tasks.size() * sizeof(TaskWord512) > kDimTargetBankBytes)
		return false;
	if ((batch.packed.row_task_ptr.size() + batch.packed.route.size())
			* sizeof(id_t) + kStats * sizeof(id_t) > kDimTargetBankBytes)
		return false;
	for (const auto& shard : batch.packed.B)
		if (shard.size() * sizeof(Beat512) > kDimTargetBankBytes) return false;
	return true;
}

static void dim_validate_rpc1_batch(const Csr& matrix,
		const DimSuperbatch& batch) {
	if (!batch.row_persistent) return;
	uint64_t physical_rows = 0;
	uint64_t completed_rows = 0;
	for (size_t command_index = 0; command_index < batch.commands.size();
			++command_index) {
		const Beat512& command = batch.commands[command_index];
		const id_t task_base = static_cast<id_t>(command.lane[0]);
		const id_t rowptr_base = static_cast<id_t>(command.lane[0] >> 32);
		const id_t route_base = static_cast<id_t>(command.lane[1]);
		const id_t rows = static_cast<id_t>(command.lane[1] >> 32);
		const id_t columns = static_cast<id_t>(command.lane[2]);
		const id_t dense_base = static_cast<id_t>(command.lane[3]);
		const bool logical_last = (command.lane[3] >> 32) & 1U;
		const uint32_t magic = static_cast<uint32_t>(command.lane[7] >> 32);
		if (magic != kRowPersistentCommandMagic || rows == 0
				|| columns == 0 || columns > kWindowColumns
				|| uint64_t(dense_base) + columns > matrix.cols
				|| uint64_t(rowptr_base) + rows
					>= batch.packed.row_task_ptr.size()
				|| uint64_t(route_base) + rows > batch.packed.route.size())
			throw std::runtime_error("invalid RPC1 command "
				+ std::to_string(command_index));
		for (id_t row = 0; row < rows; ++row) {
			const id_t task_begin
				= task_base + batch.packed.row_task_ptr[rowptr_base + row];
			const id_t task_end
				= task_base + batch.packed.row_task_ptr[rowptr_base + row + 1];
			if (task_begin > task_end || task_end > batch.task_count)
				throw std::runtime_error("RPC1 task bounds mismatch");
			for (id_t task = task_begin; task < task_end; ++task) {
				const TaskWord512& word = batch.packed.tasks[task >> 2];
				const unsigned lane = (task & 3U) * 2U;
				const uint64_t descriptor = word.lane[lane];
				const id_t offset = static_cast<id_t>(descriptor);
				const id_t length = static_cast<id_t>((descriptor >> 32) & 0xffffffU);
				const unsigned shard = (descriptor >> 56) & 7U;
				if (uint64_t(offset) + (uint64_t(length) + 7) / 8
						> batch.packed.B[shard].size())
					throw std::runtime_error("RPC1 B bounds mismatch");
				// Every source segment is sorted.  Checking its endpoints proves the
				// complete segment lies in the command page without adding another
				// O(partial-products) Host pass before every board run.
				for (unsigned endpoint = 0; endpoint < (length > 1 ? 2U : 1U);
						++endpoint) {
					const id_t item = endpoint == 0 ? 0 : length - 1;
					const Beat512& beat
						= batch.packed.B[shard][offset + item / 8];
					const id_t col = static_cast<id_t>(beat.lane[item & 7U]);
					if (col < dense_base
							|| uint64_t(col) >= uint64_t(dense_base) + columns)
						throw std::runtime_error("RPC1 global column/base mismatch");
				}
			}
		}
		physical_rows += rows;
		if (logical_last) completed_rows += rows;
	}
	if (physical_rows != batch.physical_rows
			|| completed_rows != batch.total_rows
			|| batch.logical_rows.size() != batch.total_rows)
		throw std::runtime_error("RPC1 logical/physical row accounting mismatch");
}

// Build one capacity candidate.  B payloads are packed once per physical
// column page and shared by every row command.  Commands themselves are then
// emitted in logical-row order, which is the essential difference from the
// old window-major Host.
static DimSuperbatch dim_pack_rowmajor_range(const Csr& matrix,
		const ScalableIndex& index, const adaptive_host::Plan& whole_plan,
		id_t row_begin, id_t row_end) {
	if (row_begin >= row_end || row_end > matrix.rows)
		throw std::runtime_error("invalid row-persistent row range");
	DimSuperbatch batch;
	batch.row_persistent = true;
	struct PendingWindow {
		WindowPackResult tile;
		uint32_t width = 0;
		double packing_ms = 0.0;
	};
	std::vector<PendingWindow> pending_windows;
	pending_windows.reserve(index.summary.windows);
	std::vector<DimWindowPayload> payloads;
	payloads.reserve(index.summary.windows);
	for (uint32_t window = 0; window < index.summary.windows; ++window) {
		const uint32_t width = static_cast<uint32_t>(std::min<uint64_t>(
			kWindowColumns,
			uint64_t(matrix.cols) - uint64_t(window) * kWindowColumns));
		const auto begin = Clock::now();
		WindowPackResult tile = pack_window_tile(
			matrix, index, window, row_begin, row_end);
		const double packing_ms = std::chrono::duration<double, std::milli>(
			Clock::now() - begin).count();
		pending_windows.push_back(
			{std::move(tile), width, packing_ms});
	}

	// A capacity lock is a physical legality constraint, not a new selector
	// decision.  If any 64K fragment of a complete logical row cannot execute
	// the single-pass local MERGE hierarchy, promote that complete row to DENSE
	// before applying routes to any fragment.  Promoting only the locked
	// fragment would violate the row-persistent invariant that every fragment
	// inherits one whole-row route; splitting K or materializing an intermediate
	// MERGE result would violate the single-pass architecture instead.
	adaptive_host::Plan effective_plan = whole_plan;
	uint64_t capacity_promoted_rows = 0;
	uint64_t capacity_locked_logical_rows = 0;
	for (id_t local_row = 0; local_row < row_end - row_begin; ++local_row) {
		bool locked_merge_fragment = false;
		for (const PendingWindow& pending : pending_windows) {
			const uint32_t route_word = pending.tile.packed.route[local_row];
			const id_t task_count
				= pending.tile.packed.row_task_ptr[local_row + 1]
				- pending.tile.packed.row_task_ptr[local_row];
			if (task_count > 1
					&& (route_word & kScalableCapacityLock) != 0) {
				locked_merge_fragment = true;
				break;
			}
		}
		if (locked_merge_fragment) {
			++capacity_locked_logical_rows;
			if (effective_plan.route.at(row_begin + local_row) != 3) {
				effective_plan.route[row_begin + local_row] = 3;
				++capacity_promoted_rows;
			}
			// Propagate the legality lock to every fragment of the logical row.
			// The diagnostic force-MERGE baseline therefore keeps the complete
			// row on DENSE rather than forcing only its individually legal
			// fragments back to MERGE.  DIRECT/EMPTY fragments still take their
			// normal bypass before the force-mode check in hardware.
			for (PendingWindow& pending : pending_windows)
				pending.tile.packed.route[local_row]
					|= kScalableCapacityLock;
		}
	}
	if (capacity_locked_logical_rows != 0)
		std::cout << "row_persistent_capacity_locked_logical_rows="
			<< capacity_locked_logical_rows << " row_range=[" << row_begin
			<< ',' << row_end << ")\n";
	if (capacity_promoted_rows != 0)
		std::cout << "row_persistent_capacity_promoted_rows="
			<< capacity_promoted_rows << " row_range=[" << row_begin << ','
			<< row_end << ")\n";
	for (PendingWindow& pending : pending_windows) {
		dim_apply_whole_row_routes(&pending.tile, effective_plan);
		payloads.push_back(dim_append_window_payload(&batch,
			std::move(pending.tile), pending.width, pending.packing_ms));
	}

	const id_t rows = row_end - row_begin;
	for (id_t local_row = 0; local_row < rows;) {
		std::vector<size_t> active;
		for (size_t window = 0; window < payloads.size(); ++window) {
			const auto& payload = payloads[window];
			if (payload.row_task_ptr[local_row + 1]
					!= payload.row_task_ptr[local_row])
				active.push_back(window);
		}
		// Empty and single-fragment rows with the same page can share a command.
		// This retains two-row DENSE interleaving and avoids one command per empty
		// row in very large but sparse matrices.
		const size_t single_window = active.empty() ? 0 : active.front();
		if (active.size() <= 1) {
			id_t group_end = local_row + 1;
			while (group_end < rows) {
				std::vector<size_t> next_active;
				for (size_t window = 0; window < payloads.size(); ++window)
					if (payloads[window].row_task_ptr[group_end + 1]
							!= payloads[window].row_task_ptr[group_end])
						next_active.push_back(window);
				const size_t next_window
					= next_active.empty() ? 0 : next_active.front();
				if (next_active.size() > 1 || next_window != single_window
						|| next_active.empty() != active.empty())
					break;
				++group_end;
			}
			const auto& payload = payloads[single_window];
			const id_t virtual_base
				= (batch.virtual_row_extent + 1) & ~id_t(1);
			batch.commands.push_back(dim_make_command(payload.task_base,
				payload.rowptr_base + local_row,
				payload.route_base + local_row, group_end - local_row,
				payload.width, virtual_base, payload.window, row_begin + local_row,
				static_cast<id_t>(uint64_t(payload.window) * kWindowColumns), true));
			batch.physical_rows += group_end - local_row;
			batch.virtual_row_extent = virtual_base + group_end - local_row;
			for (id_t row = local_row; row < group_end; ++row)
				batch.logical_rows.push_back(row_begin + row);
			local_row = group_end;
			continue;
		}

		for (size_t fragment = 0; fragment < active.size(); ++fragment) {
			const auto& payload = payloads[active[fragment]];
			const id_t virtual_base
				= (batch.virtual_row_extent + 1) & ~id_t(1);
			batch.commands.push_back(dim_make_command(payload.task_base,
				payload.rowptr_base + local_row,
				payload.route_base + local_row, 1, payload.width, virtual_base,
				payload.window, row_begin + local_row,
				static_cast<id_t>(uint64_t(payload.window) * kWindowColumns),
				fragment + 1 == active.size()));
			batch.physical_rows++;
			batch.virtual_row_extent = virtual_base + 1;
		}
		batch.logical_rows.push_back(row_begin + local_row);
		++local_row;
	}
	batch.total_rows = rows;
	for (id_t local_row = 0; local_row < rows; ++local_row) {
		uint64_t products = 0;
		for (const DimWindowPayload& payload : payloads)
			products += payload.route[local_row] >> kRouteModeBits;
		const uint32_t selected = effective_plan.route.at(row_begin + local_row);
		if (selected == 3) {
			batch.default_dense_rows++;
			batch.default_dense_products += products;
		} else if (selected == 2) {
			batch.default_merge_rows++;
			batch.default_merge_products += products;
		}
	}
	if (batch.packed.tasks.empty()) batch.packed.tasks.resize(1);
	return batch;
}

static std::vector<std::pair<id_t, id_t>> dim_make_rowmajor_ranges(
		const Csr& matrix, const ScalableIndex& index) {
	std::vector<uint64_t> row_tasks(matrix.rows, 0);
	std::vector<uint64_t> row_products(matrix.rows, 0);
	std::vector<uint64_t> row_output(matrix.rows, 0);
	std::vector<uint64_t> row_fragments(matrix.rows, 0);
	for (uint32_t window = 0; window < index.summary.windows; ++window) {
		const uint64_t width = std::min<uint64_t>(kWindowColumns,
			uint64_t(matrix.cols) - uint64_t(window) * kWindowColumns);
		for (const WindowRowWork& work : index.work_by_window[window]) {
			row_tasks[work.row] += work.runs;
			row_products[work.row] += work.products;
			row_output[work.row] += std::min<uint64_t>(work.products, width);
			row_fragments[work.row]++;
		}
	}
	// Products bound the unshared B payload conservatively.  Greedy eight-shard
	// placement normally makes 150M products well below 240 MiB per bank; the
	// exact post-pack check below still catches pathological imbalance.
	constexpr uint64_t kTargetProducts = 150000000;
	constexpr uint64_t kMaxCommandEstimate
		= kDimTargetBankBytes / sizeof(Beat512);
	std::vector<std::pair<id_t, id_t>> ranges;
	id_t begin = 0;
	uint64_t tasks = 0;
	uint64_t products = 0;
	uint64_t output = 0;
	uint64_t commands = 0;
	for (id_t row = 0; row < matrix.rows; ++row) {
		const uint64_t row_commands = std::max<uint64_t>(1, row_fragments[row]);
		const uint64_t next_rows = uint64_t(row) - begin + 1;
		const uint64_t metadata_bytes = next_rows * index.summary.windows
			* 2ULL * sizeof(id_t);
		if (row != begin && (tasks + row_tasks[row] > kDimTargetTasks
				|| products + row_products[row] > kTargetProducts
				|| output + row_output[row] > 120000000
				|| commands + row_commands > kMaxCommandEstimate
				|| metadata_bytes > kDimTargetBankBytes)) {
			ranges.push_back({begin, row});
			begin = row;
			tasks = products = output = commands = 0;
		}
		tasks += row_tasks[row];
		products += row_products[row];
		output += row_output[row];
		commands += row_commands;
	}
	if (begin < matrix.rows) ranges.push_back({begin, matrix.rows});
	return ranges;
}

static std::vector<std::pair<std::string, ForceMode>> dim_modes() {
	std::vector<std::pair<std::string, ForceMode>> result = {
		{"row_adaptive", kUseRoute},
		{"merge_preferred_capacity_safe", kForceMerge},
		{"dense_only", kForceDense}};
	if (const char* requested_mode = std::getenv("ADAPT_SCALABLE_MODE")) {
		const std::string requested = lower(requested_mode);
		if (requested == "all") return result;
		if (requested == "adaptive" || requested == "row_adaptive")
			return {{"row_adaptive", kUseRoute}};
		if (requested == "merge" || requested == "merge_only")
			return {{"merge_preferred_capacity_safe", kForceMerge}};
		if (requested == "dense" || requested == "dense_only")
			return {{"dense_only", kForceDense}};
		throw std::runtime_error("invalid ADAPT_SCALABLE_MODE: " + requested);
	}
	if (g_dim_row_persistent_whole_row)
		return {{std::string("host_selected_")
			+ global_mode_name(g_dim_host_selected_mode),
			force_mode_for(g_dim_host_selected_mode)}};
	return result;
}

// RPC1 emits global 32-bit columns and may concatenate several physical
// fragments before closing one CSR row.  Verify the reconstructed logical
// rows directly instead of treating each fragment as an independent tile.
static uint64_t dim_verify_row_persistent(const Csr& matrix,
		const DimSuperbatch& batch, const std::vector<id_t>& rowptr,
		const std::array<const Beat512*, 4>& C_bank,
		const std::vector<id_t>& stats) {
	if (stats.size() < kStats || stats[10] != 0 || stats[11] != 0
			|| rowptr.size() != batch.logical_rows.size() + 1
			|| rowptr.empty() || rowptr.front() != 0
			|| rowptr.back() != stats[9])
		throw std::runtime_error("row-persistent kernel returned malformed CSR");
	for (size_t row = 0; row + 1 < rowptr.size(); ++row)
		if (rowptr[row] > rowptr[row + 1] || rowptr[row + 1] > stats[9])
			throw std::runtime_error("row-persistent CSR boundary mismatch");

	const auto get_item = [&](uint64_t index) -> uint64_t {
		const uint64_t word = index >> 3;
		return C_bank[word & 3U][word >> 2].lane[index & 7U];
	};
	constexpr uint64_t kExactProducts = 10000000;
	constexpr size_t kSampleRows = 512;
	std::vector<size_t> check_rows;
	if (batch.products <= kExactProducts
			|| batch.logical_rows.size() <= kSampleRows) {
		check_rows.resize(batch.logical_rows.size());
		std::iota(check_rows.begin(), check_rows.end(), size_t{0});
	} else if (!batch.logical_rows.empty()) {
		for (size_t sample = 0; sample < kSampleRows; ++sample)
			check_rows.push_back(sample * (batch.logical_rows.size() - 1)
				/ (kSampleRows - 1));
		std::sort(check_rows.begin(), check_rows.end());
		check_rows.erase(std::unique(check_rows.begin(), check_rows.end()),
			check_rows.end());
	}

	struct Accum {
		double value = 0.0;
		double abs_sum = 0.0;
		id_t terms = 0;
	};
	for (size_t local_row : check_rows) {
		const id_t row = batch.logical_rows.at(local_row);
		std::map<id_t, Accum> expected;
		for (id_t at = matrix.rowptr[row]; at < matrix.rowptr[row + 1]; ++at) {
			const id_t k = matrix.col[at];
			for (id_t bt = matrix.rowptr[k]; bt < matrix.rowptr[k + 1]; ++bt) {
				const double product = static_cast<double>(matrix.val[at])
					* static_cast<double>(matrix.val[bt]);
				Accum& accum = expected[matrix.col[bt]];
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
			if (actual < actual_end && (actual_col >= matrix.cols
					|| (have_previous && actual_col <= previous_col)))
				throw std::runtime_error("row-persistent output order mismatch row="
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
				throw std::runtime_error("row-persistent value mismatch row="
					+ std::to_string(row) + " col="
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

static void dim_verify_mode(const Csr& matrix, const DimSuperbatch& batch,
		const std::vector<id_t>& combined_rowptr,
		const std::array<const Beat512*, 4>& C_bank,
		const std::vector<id_t>& aggregate_stats,
		DimModeTotal* total) {
	const auto get_item = [&](uint64_t index) -> uint64_t {
		const uint64_t word = index >> 3;
		return C_bank[word & 3U][word >> 2].lane[index & 7U];
	};
	if (batch.row_persistent) {
		total->sampled_rows += dim_verify_row_persistent(
			matrix, batch, combined_rowptr, C_bank, aggregate_stats);
		return;
	}
	for (const DimTileMeta& meta : batch.tiles) {
		const id_t rows = meta.row_end - meta.row_begin;
		const uint64_t output_begin = combined_rowptr[meta.output_row_base];
		const uint64_t output_end
			= combined_rowptr[meta.output_row_base + rows];
		std::vector<id_t> local_rowptr(rows + 1);
		for (id_t row = 0; row <= rows; ++row)
			local_rowptr[row]
				= combined_rowptr[meta.output_row_base + row] - output_begin;
		std::vector<uint64_t> local_items(output_end - output_begin);
		for (uint64_t index = output_begin; index < output_end; ++index)
			local_items[index - output_begin] = get_item(index);
		WindowPackResult verification_tile;
		verification_tile.window = meta.window;
		verification_tile.row_begin = meta.row_begin;
		verification_tile.row_end = meta.row_end;
		verification_tile.products = meta.products;
		verification_tile.output_nnz_upper_bound = meta.output_upper;
		std::vector<id_t> local_stats = aggregate_stats;
		local_stats[9] = local_items.size();
		local_stats[10] = 0;
		local_stats[11] = 0;
		total->sampled_rows += verify_window_tile(matrix, verification_tile,
			meta.width, local_rowptr, C_bank, local_stats, {}, &local_items);
	}
}

static void dim_execute_superbatch(const Csr& matrix,
		const DimSuperbatch& batch, xrt::device& device, xrt::kernel& kernel,
		unsigned reps, unsigned superbatch_index, std::ofstream& csv,
		std::vector<DimModeTotal>* totals) {
	const id_t capacity = std::max<uint64_t>(1, batch.output_upper);
	const uint64_t capacity_words = (uint64_t(capacity) + 7) / 8;
	const size_t words_per_bank = std::max<uint64_t>(
		1, (capacity_words + 3) / 4);
	xrt::bo command_bo(device, batch.commands.size() * sizeof(Beat512),
		kernel.group_id(0));
	xrt::bo task_bo(device, batch.packed.tasks.size() * sizeof(TaskWord512),
		kernel.group_id(1));
	xrt::bo row_bo(device,
		batch.packed.row_task_ptr.size() * sizeof(id_t), kernel.group_id(2));
	xrt::bo route_bo(device,
		batch.packed.route.size() * sizeof(id_t), kernel.group_id(3));
	std::vector<xrt::bo> B_bo;
	for (unsigned shard = 0; shard < kShards; ++shard)
		B_bo.emplace_back(device,
			batch.packed.B[shard].size() * sizeof(Beat512),
			kernel.group_id(4 + shard));
	xrt::bo C_row_bo(device, (uint64_t(batch.total_rows) + 1) * sizeof(id_t),
		kernel.group_id(23));
	std::array<xrt::bo, 4> C_item_bo = {
		xrt::bo(device, words_per_bank * sizeof(Beat512), kernel.group_id(24)),
		xrt::bo(device, words_per_bank * sizeof(Beat512), kernel.group_id(25)),
		xrt::bo(device, words_per_bank * sizeof(Beat512), kernel.group_id(26)),
		xrt::bo(device, words_per_bank * sizeof(Beat512), kernel.group_id(27))};
	xrt::bo stats_bo(device, kStats * sizeof(id_t), kernel.group_id(28));

	std::memcpy(command_bo.map<void*>(), batch.commands.data(),
		batch.commands.size() * sizeof(Beat512));
	std::memcpy(task_bo.map<void*>(), batch.packed.tasks.data(),
		batch.packed.tasks.size() * sizeof(TaskWord512));
	std::memcpy(row_bo.map<void*>(), batch.packed.row_task_ptr.data(),
		batch.packed.row_task_ptr.size() * sizeof(id_t));
	std::memcpy(route_bo.map<void*>(), batch.packed.route.data(),
		batch.packed.route.size() * sizeof(id_t));
	for (unsigned shard = 0; shard < kShards; ++shard)
		std::memcpy(B_bo[shard].map<void*>(), batch.packed.B[shard].data(),
			batch.packed.B[shard].size() * sizeof(Beat512));
	std::memset(stats_bo.map<void*>(), 0, kStats * sizeof(id_t));
	const auto h2d_begin = Clock::now();
	command_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
	task_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
	row_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
	route_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
	for (auto& bo : B_bo) bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
	stats_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
	const double h2d_ms = std::chrono::duration<double, std::milli>(
		Clock::now() - h2d_begin).count();

	const auto launch = [&](ForceMode force) {
		return kernel(command_bo, task_bo, row_bo, route_bo,
			B_bo[0], B_bo[1], B_bo[2], B_bo[3],
			B_bo[4], B_bo[5], B_bo[6], B_bo[7],
			static_cast<id_t>(batch.packed.B[0].size()),
			static_cast<id_t>(batch.packed.B[1].size()),
			static_cast<id_t>(batch.packed.B[2].size()),
			static_cast<id_t>(batch.packed.B[3].size()),
			static_cast<id_t>(batch.packed.B[4].size()),
			static_cast<id_t>(batch.packed.B[5].size()),
			static_cast<id_t>(batch.packed.B[6].size()),
			static_cast<id_t>(batch.packed.B[7].size()),
			static_cast<id_t>(batch.commands.size()),
			static_cast<uint32_t>(force), capacity, C_row_bo,
			C_item_bo[0], C_item_bo[1], C_item_bo[2], C_item_bo[3], stats_bo);
	};

	for (DimModeTotal& total : *totals) {
		auto warmup = launch(total.force);
		warmup.wait();
		std::vector<double> samples(reps, 0.0);
		for (unsigned rep = 0; rep < reps; ++rep) {
			const auto begin = Clock::now();
			auto run = launch(total.force);
			run.wait();
			samples[rep] = std::chrono::duration<double, std::milli>(
				Clock::now() - begin).count();
			total.samples[rep] += samples[rep];
		}
		const auto d2h_begin = Clock::now();
		C_row_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
		stats_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
		std::vector<id_t> rowptr(uint64_t(batch.total_rows) + 1);
		std::vector<id_t> stats(kStats);
		std::memcpy(rowptr.data(), C_row_bo.map<void*>(),
			rowptr.size() * sizeof(id_t));
		std::memcpy(stats.data(), stats_bo.map<void*>(),
			stats.size() * sizeof(id_t));
		if (stats[9] > capacity || stats[10] != 0 || stats[11] != 0)
			throw std::runtime_error("commanded kernel capacity/protocol failure");
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
		const auto verify_begin = Clock::now();
		dim_verify_mode(matrix, batch, rowptr, C_bank, stats, &total);
		const double verification_ms
			= std::chrono::duration<double, std::milli>(
				Clock::now() - verify_begin).count();
		total.h2d_ms += h2d_ms;
		total.d2h_ms += d2h_ms;
		total.verification_ms += verification_ms;
		total.output_nnz += stats[9];
		total.merge_rows += stats[3];
		total.dense_rows += stats[4];
		total.commands += batch.commands.size();
		total.rows += batch.total_rows;
		total.products += batch.products;
		total.default_merge_rows += batch.default_merge_rows;
		total.default_dense_rows += batch.default_dense_rows;
		total.default_merge_products += batch.default_merge_products;
		total.default_dense_products += batch.default_dense_products;
		total.packing_ms += batch.packing_ms;
		csv << superbatch_index << ',' << kWindowColumns << ','
			<< total.name << ','
			<< batch.commands.size() << ',' << batch.total_rows << ',' << reps
			<< ',' << batch.packing_ms << ',' << h2d_ms << ','
			<< median(samples) << ',' << d2h_ms << ',' << verification_ms
			<< ',' << batch.products << ',' << stats[9] << ',' << stats[3]
			<< ',' << stats[4] << ",1," << batch.default_merge_rows << ','
			<< batch.default_dense_rows << ',' << batch.default_merge_products
			<< ',' << batch.default_dense_products << ",-1,0\n";
		std::cout << "dim_superbatch=" << superbatch_index
			<< " mode=" << total.name
			<< " commands=" << batch.commands.size()
			<< " rows=" << batch.total_rows
			<< " kernel_ms=" << median(samples)
			<< " output_nnz=" << stats[9]
			<< " route(merge,dense)=(" << stats[3] << ',' << stats[4]
			<< ") PASS\n";
	}
}

}  // namespace

#ifndef ADAPT_DIM_HOST_MAIN
#define ADAPT_DIM_HOST_MAIN main
#endif
int ADAPT_DIM_HOST_MAIN(int argc, char** argv) {
	try {
		if (argc < 5 || argc > 6) {
			std::cerr << "usage: " << argv[0]
				<< " xclbin device matrix.mtx results.csv [reps]\n";
			return 2;
		}
		const unsigned reps = argc == 6 ? std::stoul(argv[5]) : 3;
		if (reps == 0) throw std::runtime_error("reps must be positive");
		kWindowColumns = uint32_t{1} << 16;
		std::cout << "dimension_compatibility window_capacity="
			<< kWindowColumns << " Tk=full\n";
		const Csr matrix = load_matrix_market(argv[3]);
		g_dim_row_persistent_whole_row
			= std::getenv("ADAPT_ROW_PERSISTENT_WHOLE_ROW") != nullptr;
		id_t execution_row_begin = 0;
		id_t execution_row_end = matrix.rows;
		const auto parse_execution_row = [&](const char* name,
				id_t fallback) -> id_t {
			const char* value = std::getenv(name);
			if (value == nullptr || *value == '\0') return fallback;
			const uint64_t parsed = std::stoull(value);
			if (parsed > std::numeric_limits<id_t>::max())
				throw std::runtime_error(std::string(name)
					+ " exceeds the Host row-index range");
			return static_cast<id_t>(parsed);
		};
		execution_row_begin = parse_execution_row(
			"ADAPT_EXEC_ROW_BEGIN", execution_row_begin);
		execution_row_end = parse_execution_row(
			"ADAPT_EXEC_ROW_END", execution_row_end);
		if (execution_row_begin >= execution_row_end
				|| execution_row_end > matrix.rows)
			throw std::runtime_error("invalid ADAPT_EXEC_ROW_BEGIN/END range");
		if ((execution_row_begin != 0 || execution_row_end != matrix.rows)
				&& !g_dim_row_persistent_whole_row)
			throw std::runtime_error(
				"logical-row range execution requires row-persistent mode");
		std::cout << "execution_row_range=[" << execution_row_begin << ','
			<< execution_row_end << ") selector_scope=[0," << matrix.rows
			<< ")\n";
		adaptive_host::Plan whole_row_plan;
		double selector_ms = 0.0;
		if (g_dim_row_persistent_whole_row) {
			const auto selector_begin = Clock::now();
			whole_row_plan = dim_choose_whole_row_routes(matrix);
			selector_ms = std::chrono::duration<double, std::milli>(
				Clock::now() - selector_begin).count();
			g_dim_host_selected_mode = whole_row_plan.mode;
			std::cout << "row_persistent_whole_row=1 selector_unit=logical_row "
				"fragment_order=row-major kernel_launches_due_to_N=0 selector_ms="
				<< selector_ms << '\n';
		}
		const ScalableIndex index = build_scalable_index(matrix);
		const bool dry_run = lower(argv[2]) == "dry-run";
		std::ofstream csv(argv[4]);
		if (!csv) throw std::runtime_error("cannot open dimension result CSV");
		csv << "superbatch,tn_columns,mode,commands,rows,reps,packing_ms,h2d_ms,"
			"kernel_ms,d2h_ms,verification_ms,products,output_nnz,"
			"merge_rows,dense_rows,verified,default_merge_rows,"
			"default_dense_rows,default_merge_products,"
			"default_dense_products,sample_index,selector_ms\n";

		std::vector<std::pair<std::string, ForceMode>> modes = dim_modes();
		std::vector<DimModeTotal> totals;
		for (const auto& mode : modes)
			totals.push_back({mode.first, mode.second,
				std::vector<double>(reps, 0.0)});

		std::unique_ptr<xrt::device> device;
		std::unique_ptr<xrt::kernel> kernel;
		if (!dry_run) {
			if (lower(argv[2]) == "auto" && std::getenv("TARGET_CARD") == nullptr)
				::setenv("TARGET_CARD", "u280", 0);
			device = std::make_unique<xrt::device>(
				host_device::parse_device_arg(argv[2]));
			const auto uuid = device->load_xclbin(argv[1]);
			kernel = std::make_unique<xrt::kernel>(*device, uuid,
				"adaptive_hbm_spgemm_commanded:{adaptive_hbm_spgemm_commanded_1}",
				xrt::kernel::cu_access_mode::shared);
		}

		DimSuperbatch batch;
		unsigned superbatch_index = 0;
		auto flush = [&]() {
			if (batch.commands.empty()) return;
			dim_validate_rpc1_batch(matrix, batch);
			if (dry_run) {
				std::cout << "dim_superbatch=" << superbatch_index
					<< " commands=" << batch.commands.size()
					<< " rows=" << batch.total_rows
					<< " tasks=" << batch.task_count
					<< " output_upper=" << batch.output_upper << " PACKED\n";
			} else {
				dim_execute_superbatch(matrix, batch, *device, *kernel,
					reps, superbatch_index, csv, &totals);
			}
			batch = DimSuperbatch{};
			superbatch_index++;
		};

		if (g_dim_row_persistent_whole_row
				&& matrix.cols > kWindowColumns) {
			auto ranges = dim_make_rowmajor_ranges(matrix, index);
			std::vector<std::pair<id_t, id_t>> execution_ranges;
			execution_ranges.reserve(ranges.size());
			for (const auto& range : ranges) {
				const id_t begin = std::max(range.first, execution_row_begin);
				const id_t end = std::min(range.second, execution_row_end);
				if (begin < end) execution_ranges.push_back({begin, end});
			}
			std::vector<std::pair<id_t, id_t>> pending(
				execution_ranges.rbegin(), execution_ranges.rend());
			while (!pending.empty()) {
				const auto range = pending.back();
				pending.pop_back();
				DimSuperbatch candidate = dim_pack_rowmajor_range(matrix, index,
					whole_row_plan, range.first, range.second);
				if (!dim_superbatch_fits(candidate)) {
					if (range.second - range.first <= 1)
						throw std::runtime_error("one logical row exceeds a physical "
							"HBM super-batch; split its capacity fragments across "
							"output batches");
					const id_t middle = range.first
						+ (range.second - range.first) / 2;
					pending.push_back({middle, range.second});
					pending.push_back({range.first, middle});
					continue;
				}
				batch = std::move(candidate);
				if (dry_run)
					std::cout << "row_persistent_range=[" << range.first << ','
						<< range.second << ") physical_fragments="
						<< batch.physical_rows << " commands="
						<< batch.commands.size() << '\n';
				flush();
			}
		} else for (uint32_t window = 0;
				window < index.summary.windows; ++window) {
			// The one-page case remains a single aligned multi-row command, which
			// is both faster and wire-equivalent to the general row-major path.
			const uint32_t width = static_cast<uint32_t>(std::min<uint64_t>(
				kWindowColumns,
				uint64_t(matrix.cols) - uint64_t(window) * kWindowColumns));
			auto batches = make_window_row_batches(matrix, index, window);
			if (g_dim_row_persistent_whole_row) {
				std::vector<std::pair<id_t, id_t>> execution_batches;
				execution_batches.reserve(batches.size());
				for (const auto& range : batches) {
					const id_t begin = std::max(range.first, execution_row_begin);
					const id_t end = std::min(range.second, execution_row_end);
					if (begin < end) execution_batches.push_back({begin, end});
				}
				batches = std::move(execution_batches);
			}
			if (dry_run) {
				std::cout << "dimension_window=" << window
					<< " width=" << width
					<< " row_tiles=" << batches.size();
				for (const auto& range : batches)
					std::cout << " [" << range.first << ',' << range.second << ')';
				std::cout << '\n';
			}
			for (size_t batch_index = 0; batch_index < batches.size();) {
				const auto begin = Clock::now();
				WindowPackResult tile = pack_window_tile(matrix, index, window,
					batches[batch_index].first, batches[batch_index].second);
				if (g_dim_row_persistent_whole_row) {
					// A one-page matrix has exactly one physical fragment per row,
					// so a local-MERGE capacity lock promotes that complete row to
					// DENSE directly.  The wide row-major path performs the same
					// promotion across all fragments before applying any route.
					adaptive_host::Plan effective_plan = whole_row_plan;
					uint64_t capacity_promoted_rows = 0;
					for (id_t local_row = 0;
							local_row < tile.row_end - tile.row_begin; ++local_row) {
						const uint32_t route_word = tile.packed.route[local_row];
						const id_t task_count
							= tile.packed.row_task_ptr[local_row + 1]
							- tile.packed.row_task_ptr[local_row];
						const id_t global_row = tile.row_begin + local_row;
						if (task_count > 1
								&& (route_word & kScalableCapacityLock) != 0
								&& effective_plan.route.at(global_row) != 3) {
							effective_plan.route[global_row] = 3;
							++capacity_promoted_rows;
						}
					}
					if (capacity_promoted_rows != 0)
						std::cout << "row_persistent_capacity_promoted_rows="
							<< capacity_promoted_rows << " row_range=["
							<< tile.row_begin << ',' << tile.row_end << ")\n";
					dim_apply_whole_row_routes(&tile, effective_plan);
				}
				id_t split_row = 0;
				if (find_dense_empty_shard_split(tile, true, &split_row)) {
					const auto original = batches[batch_index];
					batches[batch_index] = {original.first, split_row};
					batches.insert(batches.begin() + batch_index + 1,
						{split_row, original.second});
					continue;
				}
				const double packing_ms = std::chrono::duration<double, std::milli>(
					Clock::now() - begin).count();
				if (!dim_can_append(batch, tile)) flush();
				if (!dim_can_append(batch, tile))
					throw std::runtime_error(
						"one dimension tile exceeds super-batch capacity");
				dim_append_tile(&batch, tile, width, packing_ms);
				batch_index++;
			}
		}
		flush();

		if (dry_run) {
			std::cout << "DIMENSION_COMMAND_DRY_RUN: PASS superbatches="
				<< superbatch_index << '\n';
			return 0;
		}
		for (DimModeTotal& total : totals) {
			for (unsigned rep = 0; rep < reps; ++rep)
				csv << "SAMPLE," << kWindowColumns << ',' << total.name
					<< ",0,0," << reps << ",0,0," << total.samples[rep]
					<< ",0,0,0,0,0,0,1," << total.default_merge_rows << ','
					<< total.default_dense_rows << ','
					<< total.default_merge_products << ','
					<< total.default_dense_products << ',' << rep << ",0\n";
			csv << "TOTAL," << kWindowColumns << ',' << total.name << ','
				<< total.commands << ',' << total.rows << ',' << reps << ','
				<< total.packing_ms << ',' << total.h2d_ms << ','
				<< median(total.samples) << ',' << total.d2h_ms << ','
				<< total.verification_ms << ',' << total.products << ','
				<< total.output_nnz << ',' << total.merge_rows << ','
				<< total.dense_rows << ",1," << total.default_merge_rows << ','
				<< total.default_dense_rows << ','
				<< total.default_merge_products << ','
				<< total.default_dense_products << ",-1," << selector_ms << '\n';
			std::cout << "dimension_whole mode=" << total.name
				<< " kernel_samples_ms=[";
			for (unsigned rep = 0; rep < reps; ++rep) {
				if (rep != 0) std::cout << ',';
				std::cout << total.samples[rep];
			}
			std::cout << "] median_ms=" << median(total.samples)
				<< " output_nnz=" << total.output_nnz
				<< " route(merge,dense)=(" << total.merge_rows << ','
				<< total.dense_rows << ") sampled_rows="
				<< total.sampled_rows << '\n';
		}
		std::cout << "DIMENSION_COMMAND_BOARD: PASS superbatches="
			<< superbatch_index << '\n';
		return 0;
	} catch (const std::exception& error) {
		std::cerr << "error: " << error.what() << '\n';
		return 1;
	}
}
