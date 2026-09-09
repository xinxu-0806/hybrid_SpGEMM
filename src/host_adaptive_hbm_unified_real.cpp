// Production Host for adaptive_hbm_unified_real_hbm_tapa.
//
// Reuse the deployed Matrix Market loader, scalable B packer, complete-row
// selector and capacity checks.  The legacy commanded main is renamed so this
// translation unit can adapt its RPC1 row-major fragments to the new tagged
// four-port kernel without duplicating those correctness-critical routines.
#define ADAPT_DIM_HOST_MAIN adaptive_dim_commanded_legacy_main
#include "host_adaptive_hbm_spgemm_dim_commanded.cpp"
#undef ADAPT_DIM_HOST_MAIN

namespace {

constexpr uint64_t kRealTargetBankBytes = 240ULL << 20;
constexpr uint32_t kRealRouteEmpty = 0;
constexpr uint32_t kRealRouteMerge = 2;
constexpr uint32_t kRealRouteDense = 3;
constexpr uint32_t kRealMergeRecordRoute = 1;
constexpr uint32_t kRealDenseRecordRoute = 3;
constexpr id_t kCompletionTraceDisabled = 2;

enum class RealMode { kAdaptive, kMergePreferred, kDenseOnly };

struct RealFragmentBatch {
	std::vector<TaskWord512> tasks;
	std::vector<id_t> row_task_ptr;
	std::vector<uint32_t> base_route;
	std::vector<uint32_t> source_mask;
	std::vector<uint64_t> geometry;
	std::vector<id_t> logical_row;
	id_t task_count = 0;
	std::shared_ptr<std::array<std::vector<Beat512>, kShards> > B;
	uint64_t products = 0;
	uint64_t packet_words_upper = 0;
	uint64_t metadata_records_upper = 0;
	double packing_ms = 0.0;
};

struct RealModeTotal {
	RealMode mode = RealMode::kAdaptive;
	std::string name;
	std::vector<double> kernel_samples_ms;
	double static_h2d_ms = 0.0;
	double route_h2d_ms = 0.0;
	double d2h_ms = 0.0;
	double verification_ms = 0.0;
	uint64_t fragments = 0;
	uint64_t logical_rows = 0;
	uint64_t products = 0;
	uint64_t output_nnz = 0;
	uint64_t merge_logical_rows = 0;
	uint64_t dense_logical_rows = 0;
};

struct RealDecodedOutput {
	std::map<id_t, std::map<id_t, float> > values;
	std::map<id_t, uint64_t> eors;
	std::map<id_t, uint64_t> eor_nnz;
	uint64_t items = 0;
};

static uint64_t real_task_low(const std::vector<TaskWord512>& tasks,
		id_t task) {
	const TaskWord512& word = tasks.at(task >> 2);
	return word.lane[(task & 3U) * 2U];
}

static uint64_t real_task_high(const std::vector<TaskWord512>& tasks,
		id_t task) {
	const TaskWord512& word = tasks.at(task >> 2);
	return word.lane[(task & 3U) * 2U + 1U];
}

static uint64_t real_fragment_products(uint32_t route) {
	return (route & ~kScalableCapacityLock) >> kRouteModeBits;
}

static uint64_t real_fragment_word_upper(uint32_t route, uint64_t geometry) {
	const uint64_t span = geometry >> 32;
	const uint64_t nnz = std::min(real_fragment_products(route), span);
	uint64_t words = (nnz + 7) / 8;
	// Capacity slicing happens before a requested global mode is applied.
	// Account for the worst legal implementation here: a >32K fragment can be
	// promoted from MERGE to wide DENSE, whose two physical packetizers may each
	// finish with a partial 512-bit word.
	if (span > 32768 && nnz != 0)
		++words;  // the two 32K physical packetizers may both end partially full
	return words;
}

static void real_append_task(RealFragmentBatch* output, uint64_t low,
		uint64_t high) {
	const id_t task = output->task_count;
	const size_t word = task >> 2;
	if (output->tasks.size() <= word) output->tasks.resize(word + 1);
	const unsigned lane = (task & 3U) * 2U;
	output->tasks[word].lane[lane] = low;
	output->tasks[word].lane[lane + 1U] = high;
	++output->task_count;
}

// Build the command tail consumed independently by each B-HBM reader.
// Every tail starts with ceil(fragment_count/16) count-header beats; lane f%16
// stores the number of descriptors for fragment f on this shard.  The
// remaining beats contain four 128-bit descriptors each, concatenated in
// fragment order.  The numerical B offsets inside descriptors remain
// unchanged because the tail is appended after the original B data.
static std::array<std::vector<Beat512>, kShards>
real_build_embedded_shard_commands(const RealFragmentBatch& batch) {
	const size_t fragments = batch.logical_row.size();
	const size_t header_words = (fragments + 15) / 16;
	std::array<std::vector<Beat512>, kShards> result;
	std::array<uint64_t, kShards> descriptor_counts{};
	for (unsigned shard = 0; shard < kShards; ++shard)
		result[shard].resize(header_words);

	for (size_t fragment = 0; fragment < fragments; ++fragment) {
		std::array<uint32_t, kShards> counts{};
		for (id_t task = batch.row_task_ptr[fragment];
				task < batch.row_task_ptr[fragment + 1]; ++task) {
			const unsigned shard = (real_task_low(batch.tasks, task) >> 56) & 7U;
			if (counts[shard] == std::numeric_limits<uint32_t>::max())
				throw std::runtime_error("one fragment/shard command count overflows");
			++counts[shard];
		}
		for (unsigned shard = 0; shard < kShards; ++shard) {
			const unsigned slot = fragment & 15U;
			const unsigned lane = slot >> 1;
			const unsigned shift = (slot & 1U) * 32U;
			result[shard][fragment >> 4].lane[lane]
				|= uint64_t(counts[shard]) << shift;
			descriptor_counts[shard] += counts[shard];
			if (descriptor_counts[shard]
					> std::numeric_limits<uint32_t>::max())
				throw std::runtime_error("shard command cursor exceeds 32 bits");
		}
	}

	for (unsigned shard = 0; shard < kShards; ++shard)
		result[shard].resize(header_words
			+ (descriptor_counts[shard] + 3) / 4);
	std::array<uint64_t, kShards> cursors{};
	for (size_t fragment = 0; fragment < fragments; ++fragment) {
		for (id_t task = batch.row_task_ptr[fragment];
				task < batch.row_task_ptr[fragment + 1]; ++task) {
			const uint64_t low = real_task_low(batch.tasks, task);
			const uint64_t high = real_task_high(batch.tasks, task);
			const unsigned shard = (low >> 56) & 7U;
			const uint64_t cursor = cursors[shard]++;
			Beat512& word = result[shard][header_words + (cursor >> 2)];
			const unsigned lane = (cursor & 3U) * 2U;
			word.lane[lane] = low;
			word.lane[lane + 1U] = high;
		}
	}
	for (unsigned shard = 0; shard < kShards; ++shard)
		if (cursors[shard] != descriptor_counts[shard])
			throw std::runtime_error("embedded shard command count mismatch");
	return result;
}

static RealFragmentBatch real_expand_rpc1(DimSuperbatch* source) {
	if (!source->row_persistent)
		throw std::runtime_error("real unified Host requires RPC1 input");
	RealFragmentBatch output;
	output.B = std::make_shared<
		std::array<std::vector<Beat512>, kShards> >();
	output.row_task_ptr.push_back(0);
	size_t logical_cursor = 0;
	for (const Beat512& command : source->commands) {
		const id_t task_base = static_cast<id_t>(command.lane[0]);
		const id_t rowptr_base = static_cast<id_t>(command.lane[0] >> 32);
		const id_t route_base = static_cast<id_t>(command.lane[1]);
		const id_t rows = static_cast<id_t>(command.lane[1] >> 32);
		const id_t span = static_cast<id_t>(command.lane[2]);
		const id_t dense_base = static_cast<id_t>(command.lane[3]);
		const bool logical_last = (command.lane[3] >> 32) & 1U;
		const uint32_t magic = static_cast<uint32_t>(command.lane[7] >> 32);
		if (magic != kRowPersistentCommandMagic || rows == 0 || span == 0
				|| span > kWindowColumns)
			throw std::runtime_error("malformed RPC1 command during expansion");
		for (id_t command_row = 0; command_row < rows; ++command_row) {
			if (logical_cursor >= source->logical_rows.size())
				throw std::runtime_error("RPC1 logical row cursor overflow");
			const id_t logical = source->logical_rows[logical_cursor];
			const id_t begin = task_base
				+ source->packed.row_task_ptr.at(rowptr_base + command_row);
			const id_t end = task_base
				+ source->packed.row_task_ptr.at(rowptr_base + command_row + 1);
			uint32_t mask = 0;
			for (id_t task = begin; task < end; ++task) {
				const uint64_t low = real_task_low(source->packed.tasks, task);
				const uint64_t high = real_task_high(source->packed.tasks, task);
				mask |= uint32_t{1} << ((low >> 56) & 7U);
				real_append_task(&output, low, high);
			}
			output.row_task_ptr.push_back(output.row_task_ptr.back() + end - begin);
			const uint32_t route
				= source->packed.route.at(route_base + command_row);
			output.base_route.push_back(route);
			output.source_mask.push_back(mask);
			output.geometry.push_back(uint64_t(dense_base)
				| (uint64_t(span) << 32));
			output.logical_row.push_back(logical);
			output.products += real_fragment_products(route);
			const uint64_t words = real_fragment_word_upper(
				route, output.geometry.back());
			output.packet_words_upper += words;
			output.metadata_records_upper += words + 2;
			if (rows > 1 || logical_last) ++logical_cursor;
		}
	}
	if (logical_cursor != source->logical_rows.size()
			|| output.logical_row.size() != source->physical_rows)
		throw std::runtime_error("RPC1 physical/logical expansion mismatch");
	*output.B = std::move(source->packed.B);
	for (auto& shard : *output.B)
		if (shard.empty()) shard.resize(1);
	if (output.tasks.empty()) output.tasks.resize(1);
	output.packing_ms = source->packing_ms;
	return output;
}

static RealFragmentBatch real_slice(const RealFragmentBatch& source,
		size_t begin, size_t end) {
	if (begin >= end || end > source.logical_row.size())
		throw std::runtime_error("invalid real fragment slice");
	RealFragmentBatch output;
	output.B = source.B;
	output.row_task_ptr.push_back(0);
	for (size_t fragment = begin; fragment < end; ++fragment) {
		const id_t task_begin = source.row_task_ptr[fragment];
		const id_t task_end = source.row_task_ptr[fragment + 1];
		for (id_t task = task_begin; task < task_end; ++task)
			real_append_task(&output, real_task_low(source.tasks, task),
				real_task_high(source.tasks, task));
		output.row_task_ptr.push_back(output.row_task_ptr.back()
			+ task_end - task_begin);
		output.base_route.push_back(source.base_route[fragment]);
		output.source_mask.push_back(source.source_mask[fragment]);
		output.geometry.push_back(source.geometry[fragment]);
		output.logical_row.push_back(source.logical_row[fragment]);
		output.products += real_fragment_products(source.base_route[fragment]);
		const uint64_t words = real_fragment_word_upper(
			source.base_route[fragment], source.geometry[fragment]);
		output.packet_words_upper += words;
		output.metadata_records_upper += words + 2;
	}
	if (output.tasks.empty()) output.tasks.resize(1);
	output.packing_ms = begin == 0 ? source.packing_ms : 0.0;
	return output;
}

// Read-only packing diagnostic used to explain a deterministic board hang at
// one physical-fragment boundary.  It is deliberately Host-only and guarded
// by an environment variable, so it cannot alter the kernel ABI or production
// scheduling.  Applying it after ADAPT_DEBUG_MAX_FRAGMENTS makes the final
// records in a passing/failing prefix directly comparable.
static void real_debug_dump_fragment_tail(const RealFragmentBatch& batch) {
	const char* text = std::getenv("ADAPT_DEBUG_DUMP_FRAGMENT_TAIL");
	if (text == nullptr) return;
	const unsigned long requested = std::stoul(text);
	if (requested == 0)
		throw std::runtime_error(
			"ADAPT_DEBUG_DUMP_FRAGMENT_TAIL must be positive");
	const size_t begin = batch.logical_row.size() > requested
		? batch.logical_row.size() - requested : 0;
	for (size_t fragment = begin; fragment < batch.logical_row.size(); ++fragment) {
		const id_t task_begin = batch.row_task_ptr[fragment];
		const id_t task_end = batch.row_task_ptr[fragment + 1];
		std::array<uint64_t, kShards> shard_items{};
		std::array<uint32_t, kShards> shard_tasks{};
		uint64_t max_items = 0;
		for (id_t task = task_begin; task < task_end; ++task) {
			const uint64_t low = real_task_low(batch.tasks, task);
			const unsigned shard = (low >> 56) & 7U;
			const uint64_t items = (low >> 32) & 0xffffffU;
			shard_items[shard] += items;
			++shard_tasks[shard];
			max_items = std::max(max_items, items);
		}
		std::cout << "FRAGMENT_DEBUG index=" << fragment
			<< " logical_row=" << batch.logical_row[fragment]
			<< " same_prev=" << (fragment != 0
				&& batch.logical_row[fragment - 1] == batch.logical_row[fragment])
			<< " same_next=" << (fragment + 1 < batch.logical_row.size()
				&& batch.logical_row[fragment + 1] == batch.logical_row[fragment])
			<< " route=" << (batch.base_route[fragment] & kRouteModeMask)
			<< " capacity_lock="
			<< ((batch.base_route[fragment] & kScalableCapacityLock) != 0)
			<< " products=" << real_fragment_products(batch.base_route[fragment])
			<< " dense_base="
			<< static_cast<uint32_t>(batch.geometry[fragment])
			<< " span=" << (batch.geometry[fragment] >> 32)
			<< " source_mask=" << batch.source_mask[fragment]
			<< " tasks=" << (task_end - task_begin)
			<< " max_task_items=" << max_items;
		for (unsigned shard = 0; shard < kShards; ++shard)
			if (shard_tasks[shard] != 0)
				std::cout << " s" << shard << "=("
					<< shard_tasks[shard] << ',' << shard_items[shard] << ')';
		std::cout << '\n';
		if (std::getenv("ADAPT_DEBUG_DUMP_FRAGMENT_TASKS") != nullptr) {
			for (id_t task = task_begin; task < task_end; ++task) {
				const uint64_t low = real_task_low(batch.tasks, task);
				const uint64_t high = real_task_high(batch.tasks, task);
				std::cout << "FRAGMENT_TASK fragment=" << fragment
					<< " local_task=" << (task - task_begin)
					<< " shard=" << ((low >> 56) & 7U)
					<< " offset=" << static_cast<uint32_t>(low)
					<< " items=" << ((low >> 32) & 0xffffffU)
					<< " packed_route=" << ((low >> 59) & 7U)
					<< " row_last=" << ((low >> 63) & 1U)
					<< " high=" << high << '\n';
			}
		}
	}
}

// Diagnostic-only command filter.  It is used on an already isolated row to
// determine whether one shard's Nth B-row command triggers a protocol hang.
// Numerical verification is expected to fail after a truncated experiment;
// the useful observation is whether both kernel launches return.  Production
// runs explicitly clear this environment variable.
static void real_debug_limit_tasks_per_shard(RealFragmentBatch* batch) {
	const char* limit_text = std::getenv("ADAPT_DEBUG_MAX_TASKS_PER_SHARD");
	const char* drop_text = std::getenv("ADAPT_DEBUG_DROP_FRAGMENT_TASK");
	if (limit_text == nullptr && drop_text == nullptr) return;
	const unsigned long limit = limit_text == nullptr
		? std::numeric_limits<unsigned long>::max() : std::stoul(limit_text);
	if (limit == 0)
		throw std::runtime_error(
			"ADAPT_DEBUG_MAX_TASKS_PER_SHARD must be positive");
	size_t drop_fragment = std::numeric_limits<size_t>::max();
	unsigned long drop_task = std::numeric_limits<unsigned long>::max();
	if (drop_text != nullptr) {
		const std::string drop(drop_text);
		const size_t separator = drop.find(':');
		if (separator == std::string::npos)
			throw std::runtime_error(
				"ADAPT_DEBUG_DROP_FRAGMENT_TASK must be fragment:local_task");
		drop_fragment = std::stoul(drop.substr(0, separator));
		drop_task = std::stoul(drop.substr(separator + 1));
	}
	RealFragmentBatch filtered;
	filtered.B = batch->B;
	filtered.row_task_ptr.push_back(0);
	for (size_t fragment = 0; fragment < batch->logical_row.size(); ++fragment) {
		std::array<unsigned long, kShards> kept{};
		uint32_t source_mask = 0;
		for (id_t task = batch->row_task_ptr[fragment];
				task < batch->row_task_ptr[fragment + 1]; ++task) {
			const unsigned long local_task
				= task - batch->row_task_ptr[fragment];
			if (fragment == drop_fragment && local_task == drop_task) continue;
			const uint64_t low = real_task_low(batch->tasks, task);
			const unsigned shard = (low >> 56) & 7U;
			if (kept[shard] >= limit) continue;
			++kept[shard];
			source_mask |= uint32_t{1} << shard;
			real_append_task(&filtered, low, real_task_high(batch->tasks, task));
		}
		filtered.row_task_ptr.push_back(filtered.task_count);
		filtered.base_route.push_back(batch->base_route[fragment]);
		filtered.source_mask.push_back(source_mask);
		filtered.geometry.push_back(batch->geometry[fragment]);
		filtered.logical_row.push_back(batch->logical_row[fragment]);
	}
	if (filtered.tasks.empty()) filtered.tasks.resize(1);
	filtered.products = batch->products;
	filtered.packet_words_upper = batch->packet_words_upper;
	filtered.metadata_records_upper = batch->metadata_records_upper;
	filtered.packing_ms = batch->packing_ms;
	*batch = std::move(filtered);
	if (limit_text != nullptr)
		std::cout << "[HOST] debug max tasks per shard=" << limit << std::endl;
	if (drop_text != nullptr)
		std::cout << "[HOST] debug dropped fragment task=" << drop_text
			<< std::endl;
}

// Diagnostic probe for dispatcher head-of-line blocking.  Preserve each
// shard's command order, but interleave the eight shard-local queues one task
// per round.  MERGE sees the same sorted runs in the same per-shard order and
// DENSE is order-independent, so this transformation is numerically neutral.
// Once confirmed on hardware, the same ordering becomes part of production
// packing together with early per-shard boundaries.
static void real_debug_round_robin_tasks(RealFragmentBatch* batch) {
	if (std::getenv("ADAPT_DEBUG_ROUND_ROBIN_TASKS") == nullptr) return;
	RealFragmentBatch reordered;
	reordered.B = batch->B;
	reordered.row_task_ptr.push_back(0);
	for (size_t fragment = 0; fragment < batch->logical_row.size(); ++fragment) {
		std::array<std::vector<std::pair<uint64_t, uint64_t> >, kShards> queues;
		size_t rounds = 0;
		for (id_t task = batch->row_task_ptr[fragment];
				task < batch->row_task_ptr[fragment + 1]; ++task) {
			const uint64_t low = real_task_low(batch->tasks, task);
			const unsigned shard = (low >> 56) & 7U;
			queues[shard].push_back({low, real_task_high(batch->tasks, task)});
			rounds = std::max(rounds, queues[shard].size());
		}
		for (size_t round = 0; round < rounds; ++round)
			for (unsigned shard = 0; shard < kShards; ++shard)
				if (round < queues[shard].size())
					real_append_task(&reordered, queues[shard][round].first,
						queues[shard][round].second);
		reordered.row_task_ptr.push_back(reordered.task_count);
		reordered.base_route.push_back(batch->base_route[fragment]);
		reordered.source_mask.push_back(batch->source_mask[fragment]);
		reordered.geometry.push_back(batch->geometry[fragment]);
		reordered.logical_row.push_back(batch->logical_row[fragment]);
	}
	if (reordered.tasks.empty()) reordered.tasks.resize(1);
	reordered.products = batch->products;
	reordered.packet_words_upper = batch->packet_words_upper;
	reordered.metadata_records_upper = batch->metadata_records_upper;
	reordered.packing_ms = batch->packing_ms;
	*batch = std::move(reordered);
	std::cout << "[HOST] debug round-robin shard tasks=1" << std::endl;
}

static std::vector<RealFragmentBatch> real_capacity_slices(
		const RealFragmentBatch& source) {
	const uint64_t max_words = kRealTargetBankBytes / sizeof(Beat512);
	std::vector<RealFragmentBatch> result;
	size_t begin = 0;
	uint64_t words = 0;
	for (size_t fragment = 0; fragment < source.logical_row.size(); ++fragment) {
		const uint64_t next = real_fragment_word_upper(
			source.base_route[fragment], source.geometry[fragment]);
		if (fragment != begin && words + next > max_words) {
			result.push_back(real_slice(source, begin, fragment));
			begin = fragment;
			words = 0;
		}
		if (next > max_words)
			throw std::runtime_error("one <=64K fragment exceeds output HBM bank");
		words += next;
	}
	if (begin < source.logical_row.size())
		result.push_back(real_slice(source, begin, source.logical_row.size()));
	return result;
}

static std::string real_mode_name(RealMode mode) {
	switch (mode) {
	case RealMode::kAdaptive: return "row_adaptive";
	case RealMode::kMergePreferred: return "merge_preferred_capacity_safe";
	case RealMode::kDenseOnly: return "dense_only";
	}
	return "invalid";
}

static std::vector<RealMode> real_requested_modes() {
	const char* value = std::getenv("ADAPT_SCALABLE_MODE");
	if (value == nullptr || lower(value) == "all")
		return {RealMode::kAdaptive, RealMode::kMergePreferred,
			RealMode::kDenseOnly};
	const std::string mode = lower(value);
	if (mode == "adaptive" || mode == "row_adaptive")
		return {RealMode::kAdaptive};
	if (mode == "merge" || mode == "merge_only")
		return {RealMode::kMergePreferred};
	if (mode == "dense" || mode == "dense_only")
		return {RealMode::kDenseOnly};
	throw std::runtime_error("invalid ADAPT_SCALABLE_MODE: " + mode);
}

// Offline admission audit for the frozen paper set.  It deliberately uses the
// exact packed batch that would be sent to XRT, but it has no effect on the
// production launch path.  Aggregate capacities alone can hide one hot HBM
// shard or one pathological fragment, so report both maxima explicitly.
static void real_print_dry_run_diagnostics(const RealFragmentBatch& batch,
		unsigned batch_index) {
	const auto embedded_commands = real_build_embedded_shard_commands(batch);
	uint64_t b_total_words = 0, b_max_words = 0;
	uint64_t command_total_words = 0, command_max_words = 0;
	uint64_t combined_max_words = 0;
	for (unsigned shard = 0; shard < kShards; ++shard) {
		b_total_words += batch.B->at(shard).size();
		b_max_words = std::max<uint64_t>(
			b_max_words, batch.B->at(shard).size());
		command_total_words += embedded_commands[shard].size();
		command_max_words = std::max<uint64_t>(
			command_max_words, embedded_commands[shard].size());
		combined_max_words = std::max<uint64_t>(combined_max_words,
			batch.B->at(shard).size() + embedded_commands[shard].size());
	}
	if (combined_max_words * sizeof(Beat512) > kRealTargetBankBytes)
		throw std::runtime_error(
			"dry-run B data plus embedded commands exceeds one HBM bank");
	uint64_t max_products = 0, max_words = 0;
	id_t max_tasks = 0;
	id_t max_tasks_one_shard = 0;
	uint64_t wide_dense = 0, merge_fragments = 0, dense_fragments = 0;
	for (size_t fragment = 0; fragment < batch.logical_row.size(); ++fragment) {
		std::array<id_t, kShards> shard_tasks{};
		for (id_t task = batch.row_task_ptr[fragment];
				task < batch.row_task_ptr[fragment + 1]; ++task) {
			const unsigned shard = (real_task_low(batch.tasks, task) >> 56) & 7U;
			max_tasks_one_shard = std::max(
				max_tasks_one_shard, ++shard_tasks[shard]);
		}
		const uint32_t route = batch.base_route[fragment] & kRouteModeMask;
		if (route == kRealRouteDense) {
			++dense_fragments;
			if ((batch.geometry[fragment] >> 32) > 32768) ++wide_dense;
		} else {
			++merge_fragments;
		}
		max_products = std::max<uint64_t>(max_products,
			real_fragment_products(batch.base_route[fragment]));
		max_words = std::max<uint64_t>(max_words,
			real_fragment_word_upper(batch.base_route[fragment],
				batch.geometry[fragment]));
		max_tasks = std::max<id_t>(max_tasks,
			batch.row_task_ptr[fragment + 1] - batch.row_task_ptr[fragment]);
	}
	std::cout << "real_batch_audit=" << batch_index
		<< " B_words(total,max_shard)=(" << b_total_words << ',' << b_max_words
		<< ") embedded_command_words(total,max_shard)=("
		<< command_total_words << ',' << command_max_words
		<< ") combined_max_shard_words=" << combined_max_words
		<< " tasks=" << batch.task_count << " max_tasks=" << max_tasks
		<< " max_tasks_one_shard=" << max_tasks_one_shard
		<< " max_products=" << max_products << " max_output_words=" << max_words
		<< " route_fragments(M,D)=(" << merge_fragments << ',' << dense_fragments
		<< ") wide_dense=" << wide_dense << std::endl;
}

static std::vector<uint32_t> real_routes_for_mode(
		const RealFragmentBatch& batch, RealMode mode,
		uint64_t* merge_rows, uint64_t* dense_rows) {
	std::vector<uint32_t> routes(batch.base_route.size());
	std::map<id_t, uint32_t> logical_modes;
	for (size_t fragment = 0; fragment < routes.size(); ++fragment) {
		const uint32_t base = batch.base_route[fragment];
		const bool nonempty
			= batch.row_task_ptr[fragment] != batch.row_task_ptr[fragment + 1];
		uint32_t selected = base & kRouteModeMask;
		if (!nonempty) selected = kRealRouteEmpty;
		else if (mode == RealMode::kDenseOnly) selected = kRealRouteDense;
		else if (mode == RealMode::kMergePreferred)
			selected = (base & kScalableCapacityLock) != 0
				? kRealRouteDense : kRealRouteMerge;
		routes[fragment] = (base & ~kRouteModeMask) | selected;
		if (selected == kRealRouteDense)
			logical_modes[batch.logical_row[fragment]] = kRealRouteDense;
		else if (selected != kRealRouteEmpty
				&& !logical_modes.count(batch.logical_row[fragment]))
			logical_modes[batch.logical_row[fragment]] = kRealRouteMerge;
	}
	*merge_rows = 0;
	*dense_rows = 0;
	for (const auto& entry : logical_modes) {
		if (entry.second == kRealRouteDense) ++*dense_rows;
		else ++*merge_rows;
	}
	return routes;
}

static bool real_column_in_batch(const RealFragmentBatch& batch,
		id_t row, id_t column) {
	for (size_t fragment = 0; fragment < batch.logical_row.size(); ++fragment) {
		if (batch.logical_row[fragment] != row) continue;
		const uint64_t base = static_cast<uint32_t>(batch.geometry[fragment]);
		const uint64_t span = batch.geometry[fragment] >> 32;
		if (column >= base && uint64_t(column) < base + span) return true;
	}
	return false;
}

static uint64_t real_verify(const Csr& matrix,
		const RealFragmentBatch& batch, const RealDecodedOutput& observed) {
	std::vector<id_t> rows = batch.logical_row;
	std::sort(rows.begin(), rows.end());
	rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
	constexpr uint64_t kExactProducts = 10000000;
	constexpr size_t kSampleRows = 512;
	if (batch.products > kExactProducts && rows.size() > kSampleRows) {
		std::vector<id_t> sample;
		for (size_t index = 0; index < kSampleRows; ++index)
			sample.push_back(rows[index * (rows.size() - 1) / (kSampleRows - 1)]);
		std::sort(sample.begin(), sample.end());
		sample.erase(std::unique(sample.begin(), sample.end()), sample.end());
		rows = std::move(sample);
	}
	struct Accum { double value = 0; double abs_sum = 0; id_t terms = 0; };
	for (id_t row : rows) {
		std::map<id_t, Accum> expected;
		for (id_t at = matrix.rowptr[row]; at < matrix.rowptr[row + 1]; ++at) {
			const id_t k = matrix.col[at];
			for (id_t bt = matrix.rowptr[k]; bt < matrix.rowptr[k + 1]; ++bt) {
				const id_t col = matrix.col[bt];
				if (!real_column_in_batch(batch, row, col)) continue;
				const double product = static_cast<double>(matrix.val[at])
					* static_cast<double>(matrix.val[bt]);
				Accum& value = expected[col];
				value.value += product;
				value.abs_sum += std::fabs(product);
				++value.terms;
			}
		}
		const auto actual_row = observed.values.find(row);
		const std::map<id_t, float> empty;
		const auto& actual = actual_row == observed.values.end()
			? empty : actual_row->second;
		auto want = expected.begin();
		auto got = actual.begin();
		while (want != expected.end() || got != actual.end()) {
			const id_t want_col = want == expected.end()
				? std::numeric_limits<id_t>::max() : want->first;
			const id_t got_col = got == actual.end()
				? std::numeric_limits<id_t>::max() : got->first;
			double expected_value = 0.0;
			double abs_sum = 0.0;
			id_t terms = 0;
			double actual_value = 0.0;
			if (want_col <= got_col) {
				expected_value = want->second.value;
				abs_sum = want->second.abs_sum;
				terms = want->second.terms;
			}
			if (got_col <= want_col) actual_value = got->second;
			const double roundoff = (2.0 * terms + 8.0)
				* std::ldexp(1.0, -24);
			const double gamma = roundoff < 1.0
				? roundoff / (1.0 - roundoff)
				: std::numeric_limits<double>::infinity();
			const double tolerance = std::max(
				1.0e-4 + 2.0e-4 * std::fabs(expected_value), gamma * abs_sum);
			if (!std::isfinite(actual_value)
					|| std::fabs(actual_value - expected_value) > tolerance)
				throw std::runtime_error("real unified value mismatch row="
					+ std::to_string(row) + " col="
					+ std::to_string(std::min(want_col, got_col)));
			if (want_col <= got_col) ++want;
			if (got_col <= want_col) ++got;
		}
	}
	return rows.size();
}

static RealDecodedOutput real_decode_output(
		const RealFragmentBatch& batch, const std::vector<uint32_t>& routes,
		const std::array<const Beat512*, 4>& data,
		const Beat512* metadata, const std::array<std::vector<id_t>, 4>& port_stats,
		const std::vector<id_t>& meta_stats,
		const std::vector<uint64_t>& completions,
		const std::vector<id_t>& completion_stats) {
	std::map<id_t, uint64_t> expected_eors;
	std::map<id_t, uint64_t> expected_completions;
	std::map<id_t, uint32_t> expected_record_route;
	for (size_t fragment = 0; fragment < batch.logical_row.size(); ++fragment) {
		const id_t row = batch.logical_row[fragment];
		const uint32_t route = routes[fragment] & kRouteModeMask;
		expected_eors[row]++;
		if (route == kRealRouteDense
				&& (batch.geometry[fragment] >> 32) > 32768)
			expected_eors[row]++;
		expected_completions[row]++;
		expected_record_route[row] = route == kRealRouteDense
			? kRealDenseRecordRoute : kRealMergeRecordRoute;
	}
	RealDecodedOutput output;
	std::array<uint64_t, 4> consumed{};
	for (uint64_t word = 0; word < meta_stats.at(0); ++word) {
		for (unsigned slot = 0; slot < 4; ++slot) {
			const uint64_t low = metadata[word].lane[slot * 2];
			const uint64_t high = metadata[word].lane[slot * 2 + 1];
			const bool present = (high >> 63) & 1U;
			if (!present) continue;
			const id_t row = static_cast<id_t>(low);
			const id_t row_nnz = static_cast<id_t>(low >> 32);
			const unsigned valid = high & 0xfU;
			const bool eor = (high >> 4) & 1U;
			const unsigned route = (high >> 8) & 3U;
			const unsigned error = (high >> 14) & 3U;
			const unsigned port = (high >> 16) & 3U;
			if (!expected_record_route.count(row)
					|| expected_record_route[row] != route || error != 0 || port > 3)
				throw std::runtime_error("real unified malformed metadata record");
			if (valid != 0) {
				if (consumed[port] >= static_cast<uint64_t>(port_stats[port][0]))
					throw std::runtime_error("real unified data/meta cursor mismatch");
				const Beat512& packet = data[port][consumed[port]++];
				for (unsigned lane = 0; lane < valid; ++lane) {
					const uint64_t item = packet.lane[lane];
					const id_t col = static_cast<id_t>(item);
					if (!real_column_in_batch(batch, row, col))
						throw std::runtime_error("real unified column outside fragment");
					uint32_t bits = static_cast<uint32_t>(item >> 32);
					float value;
					std::memcpy(&value, &bits, sizeof(value));
					auto inserted = output.values[row].emplace(col, value);
					if (!inserted.second)
						throw std::runtime_error("duplicate output column across fragments");
					++output.items;
				}
			}
			if (eor) {
				++output.eors[row];
				output.eor_nnz[row] += row_nnz;
			}
		}
	}
	for (unsigned port = 0; port < 4; ++port)
		if (consumed[port] != static_cast<uint64_t>(port_stats[port][0]))
			throw std::runtime_error("unconsumed real unified output word");
	for (const auto& entry : expected_eors)
		if (output.eors[entry.first] != entry.second)
			throw std::runtime_error("real unified EOR count mismatch");
	if (completion_stats.at(1) != kCompletionTraceDisabled) {
		std::map<id_t, uint64_t> got_completions;
		for (id_t index = 0; index < completion_stats.at(0); ++index) {
			const uint64_t event = completions[index];
			if ((event >> 49) & 1U)
				throw std::runtime_error("real unified wide completion mismatch");
			++got_completions[(event >> 11) & 0xffffffffU];
		}
		if (got_completions != expected_completions)
			throw std::runtime_error("real unified completion count mismatch");
	}
	return output;
}

static void real_execute_batch(const Csr& matrix,
		const RealFragmentBatch& batch, xrt::device& device, xrt::kernel& kernel,
		unsigned reps, unsigned batch_index, id_t heavy_threshold,
		std::ofstream& csv, std::vector<RealModeTotal>* totals) {
	const id_t fragments = batch.logical_row.size();
	id_t data_capacity = std::max<uint64_t>(1, batch.packet_words_upper);
	if (const char* capacity_text = std::getenv("ADAPT_DEBUG_DATA_WORD_CAPACITY")) {
		const unsigned long requested = std::stoul(capacity_text);
		if (requested == 0)
			throw std::runtime_error("ADAPT_DEBUG_DATA_WORD_CAPACITY must be positive");
		data_capacity = requested;
		std::cout << "[HOST] debug data-word capacity=" << data_capacity
			<< std::endl;
	}
	const id_t meta_capacity = std::max<uint64_t>(1,
		(batch.metadata_records_upper + 3) / 4);
	const id_t completion_capacity = std::max<id_t>(1, fragments);
	if (uint64_t(data_capacity) * sizeof(Beat512) > kRealTargetBankBytes)
		throw std::runtime_error("real unified output slice exceeds one HBM bank");
	// Board-side progress markers distinguish BO setup, HBM DMA and kernel wait
	// when a long real workload stops after the small protocol gates pass.
	std::cout << "[HOST] batch=" << batch_index << " prepare fragments="
		<< fragments << " data_words=" << data_capacity << std::endl;

	xrt::bo route_bo(device, batch.base_route.size() * sizeof(uint32_t),
		kernel.group_id(0));
	xrt::bo mask_bo(device, batch.source_mask.size() * sizeof(uint32_t),
		kernel.group_id(1));
	xrt::bo geometry_bo(device, batch.geometry.size() * sizeof(uint64_t),
		kernel.group_id(2));
	xrt::bo logical_bo(device, batch.logical_row.size() * sizeof(id_t),
		kernel.group_id(3));
	const auto embedded_commands = real_build_embedded_shard_commands(batch);
	std::vector<xrt::bo> B_bo;
	B_bo.reserve(kShards);
	for (unsigned shard = 0; shard < kShards; ++shard) {
		const uint64_t words = batch.B->at(shard).size()
			+ embedded_commands[shard].size();
		if (words * sizeof(Beat512) > kRealTargetBankBytes)
			throw std::runtime_error(
				"B data plus embedded shard commands exceeds one HBM bank");
		B_bo.emplace_back(device, words * sizeof(Beat512),
			kernel.group_id(4 + shard));
	}
	std::array<xrt::bo, 4> output_bo = {
		xrt::bo(device, uint64_t(data_capacity) * sizeof(Beat512), kernel.group_id(25)),
		xrt::bo(device, uint64_t(data_capacity) * sizeof(Beat512), kernel.group_id(26)),
		xrt::bo(device, uint64_t(data_capacity) * sizeof(Beat512), kernel.group_id(27)),
		xrt::bo(device, uint64_t(data_capacity) * sizeof(Beat512), kernel.group_id(28))};
	xrt::bo metadata_bo(device, uint64_t(meta_capacity) * sizeof(Beat512),
		kernel.group_id(29));
	xrt::bo completion_bo(device,
		uint64_t(completion_capacity) * sizeof(uint64_t), kernel.group_id(30));
	std::array<xrt::bo, 4> port_stats_bo = {
		xrt::bo(device, 4 * sizeof(id_t), kernel.group_id(31)),
		xrt::bo(device, 4 * sizeof(id_t), kernel.group_id(32)),
		xrt::bo(device, 4 * sizeof(id_t), kernel.group_id(33)),
		xrt::bo(device, 4 * sizeof(id_t), kernel.group_id(34))};
	xrt::bo meta_stats_bo(device, 4 * sizeof(id_t), kernel.group_id(35));
	xrt::bo completion_stats_bo(device, 2 * sizeof(id_t), kernel.group_id(36));
	xrt::bo scheduler_stats_bo(device, 16 * sizeof(id_t), kernel.group_id(37));
	xrt::bo dense_stats_bo(device, 12 * sizeof(id_t), kernel.group_id(38));
	xrt::bo heavy_stats_bo(device, 8 * sizeof(id_t), kernel.group_id(39));
	std::cout << "[HOST] batch=" << batch_index << " bo_allocate complete"
		<< std::endl;

	std::memcpy(mask_bo.map<void*>(), batch.source_mask.data(),
		batch.source_mask.size() * sizeof(uint32_t));
	std::memcpy(geometry_bo.map<void*>(), batch.geometry.data(),
		batch.geometry.size() * sizeof(uint64_t));
	std::memcpy(logical_bo.map<void*>(), batch.logical_row.data(),
		batch.logical_row.size() * sizeof(id_t));
	for (unsigned shard = 0; shard < kShards; ++shard) {
		Beat512* destination = B_bo[shard].map<Beat512*>();
		std::memcpy(destination, batch.B->at(shard).data(),
			batch.B->at(shard).size() * sizeof(Beat512));
		std::memcpy(destination + batch.B->at(shard).size(),
			embedded_commands[shard].data(),
			embedded_commands[shard].size() * sizeof(Beat512));
	}
	const auto static_h2d_begin = Clock::now();
	mask_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
	geometry_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
	logical_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
	for (xrt::bo& bo : B_bo) bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
	const double static_h2d_ms = std::chrono::duration<double, std::milli>(
		Clock::now() - static_h2d_begin).count();
	std::cout << "[HOST] batch=" << batch_index << " static_h2d complete ms="
		<< static_h2d_ms << std::endl;

	for (RealModeTotal& total : *totals) {
		uint64_t merge_rows = 0, dense_rows = 0;
		const std::vector<uint32_t> routes = real_routes_for_mode(
			batch, total.mode, &merge_rows, &dense_rows);
		std::memcpy(route_bo.map<void*>(), routes.data(),
			routes.size() * sizeof(uint32_t));
		const auto route_h2d_begin = Clock::now();
		route_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
		const double route_h2d_ms = std::chrono::duration<double, std::milli>(
			Clock::now() - route_h2d_begin).count();
		std::cout << "[HOST] batch=" << batch_index << " mode=" << total.name
			<< " route_h2d complete ms=" << route_h2d_ms << std::endl;
		const auto launch = [&]() {
			return kernel(route_bo, mask_bo, geometry_bo,
				logical_bo, B_bo[0], B_bo[1], B_bo[2], B_bo[3], B_bo[4],
				B_bo[5], B_bo[6], B_bo[7],
				static_cast<id_t>(batch.B->at(0).size()),
				static_cast<id_t>(batch.B->at(1).size()),
				static_cast<id_t>(batch.B->at(2).size()),
				static_cast<id_t>(batch.B->at(3).size()),
				static_cast<id_t>(batch.B->at(4).size()),
				static_cast<id_t>(batch.B->at(5).size()),
				static_cast<id_t>(batch.B->at(6).size()),
				static_cast<id_t>(batch.B->at(7).size()),
				fragments, heavy_threshold, data_capacity, meta_capacity,
				completion_capacity, output_bo[0], output_bo[1], output_bo[2],
				output_bo[3], metadata_bo, completion_bo, port_stats_bo[0],
				port_stats_bo[1], port_stats_bo[2], port_stats_bo[3],
				meta_stats_bo, completion_stats_bo, scheduler_stats_bo,
				dense_stats_bo, heavy_stats_bo);
		};
		std::cout << "[HOST] batch=" << batch_index << " mode=" << total.name
			<< " warmup launch" << std::endl;
		auto warmup = launch();
		warmup.wait();
		std::cout << "[HOST] batch=" << batch_index << " mode=" << total.name
			<< " warmup complete" << std::endl;
		std::vector<double> batch_samples(reps);
		for (unsigned rep = 0; rep < reps; ++rep) {
			const auto begin = Clock::now();
			auto run = launch();
			run.wait();
			batch_samples[rep] = std::chrono::duration<double, std::milli>(
				Clock::now() - begin).count();
			total.kernel_samples_ms[rep] += batch_samples[rep];
		}
		const auto d2h_begin = Clock::now();
		for (xrt::bo& bo : port_stats_bo) bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
		meta_stats_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
		completion_stats_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
		scheduler_stats_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
		dense_stats_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
		heavy_stats_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
		std::array<std::vector<id_t>, 4> port_stats;
		for (unsigned port = 0; port < 4; ++port) {
			port_stats[port].resize(4);
			std::memcpy(port_stats[port].data(), port_stats_bo[port].map<void*>(),
				4 * sizeof(id_t));
			if (port_stats[port][3] != 0
					|| port_stats[port][0] > data_capacity)
				throw std::runtime_error("real unified output port overflow");
			if (port_stats[port][0] != 0)
				output_bo[port].sync(XCL_BO_SYNC_BO_FROM_DEVICE,
					uint64_t(port_stats[port][0]) * sizeof(Beat512), 0);
		}
		std::vector<id_t> meta_stats(4), completion_stats(2), scheduler_stats(16);
		std::memcpy(meta_stats.data(), meta_stats_bo.map<void*>(),
			meta_stats.size() * sizeof(id_t));
		std::memcpy(completion_stats.data(), completion_stats_bo.map<void*>(),
			completion_stats.size() * sizeof(id_t));
		std::memcpy(scheduler_stats.data(), scheduler_stats_bo.map<void*>(),
			scheduler_stats.size() * sizeof(id_t));
		if (meta_stats[3] != 0
				|| (completion_stats[1] != 0
					&& completion_stats[1] != kCompletionTraceDisabled)
				|| meta_stats[0] > meta_capacity
				|| completion_stats[0] > completion_capacity
				|| scheduler_stats[0] != fragments
				|| scheduler_stats[1] != fragments || scheduler_stats[13] != 0)
			throw std::runtime_error("real unified scheduler/metadata failure");
		if (meta_stats[0] != 0)
			metadata_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE,
				uint64_t(meta_stats[0]) * sizeof(Beat512), 0);
		if (completion_stats[1] != kCompletionTraceDisabled
				&& completion_stats[0] != 0)
			completion_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE,
				uint64_t(completion_stats[0]) * sizeof(uint64_t), 0);
		const double d2h_ms = std::chrono::duration<double, std::milli>(
			Clock::now() - d2h_begin).count();
		const std::array<const Beat512*, 4> output_data = {
			output_bo[0].map<const Beat512*>(), output_bo[1].map<const Beat512*>(),
			output_bo[2].map<const Beat512*>(), output_bo[3].map<const Beat512*>()};
		std::vector<uint64_t> completions(completion_stats[0]);
		if (!completions.empty())
			std::memcpy(completions.data(), completion_bo.map<void*>(),
				completions.size() * sizeof(uint64_t));
		const auto verify_begin = Clock::now();
		const RealDecodedOutput decoded = real_decode_output(batch, routes,
			output_data, metadata_bo.map<const Beat512*>(), port_stats,
			meta_stats, completions, completion_stats);
		const uint64_t verified_rows = real_verify(matrix, batch, decoded);
		const double verification_ms = std::chrono::duration<double, std::milli>(
			Clock::now() - verify_begin).count();

		total.static_h2d_ms += static_h2d_ms;
		total.route_h2d_ms += route_h2d_ms;
		total.d2h_ms += d2h_ms;
		total.verification_ms += verification_ms;
		total.fragments += fragments;
		total.logical_rows += verified_rows;
		total.products += batch.products;
		total.output_nnz += decoded.items;
		total.merge_logical_rows += merge_rows;
		total.dense_logical_rows += dense_rows;
		csv << batch_index << ',' << total.name << ',' << fragments << ','
			<< merge_rows << ',' << dense_rows << ',' << batch.products << ','
			<< decoded.items << ',' << batch.packing_ms << ',' << static_h2d_ms
			<< ',' << route_h2d_ms << ',' << median(batch_samples) << ','
			<< d2h_ms << ',' << verification_ms << ',' << verified_rows << '\n';
		std::cout << "real_batch=" << batch_index << " mode=" << total.name
			<< " fragments=" << fragments << " route(M,D)=(" << merge_rows
			<< ',' << dense_rows << ") kernel_ms=" << median(batch_samples)
			<< " output_nnz=" << decoded.items << " PASS\n";
	}
}

}  // namespace

int main(int argc, char** argv) {
	try {
		if (argc < 5 || argc > 6) {
			std::cerr << "usage: " << argv[0]
				<< " xclbin device matrix.mtx results.csv [reps]\n";
			return 2;
		}
		const unsigned reps = argc == 6 ? std::stoul(argv[5]) : 10;
		if (reps == 0) throw std::runtime_error("reps must be positive");
		kWindowColumns = uint32_t{1} << 16;
		g_dim_row_persistent_whole_row = true;
		const Csr matrix = load_matrix_market(argv[3]);
		if (matrix.rows != matrix.cols)
			throw std::runtime_error("paper protocol requires square A x A");
		const auto selector_begin = Clock::now();
		const adaptive_host::Plan whole_plan = dim_choose_whole_row_routes(matrix);
		const double selector_ms = std::chrono::duration<double, std::milli>(
			Clock::now() - selector_begin).count();
		const ScalableIndex index = build_scalable_index(matrix);
		const bool dry_run = lower(argv[2]) == "dry-run";
		id_t heavy_threshold = 4096;
		if (const char* value = std::getenv("ADAPT_HEAVY_MERGE_PRODUCTS"))
			heavy_threshold = static_cast<id_t>(std::stoul(value));

		std::ofstream csv(argv[4]);
		if (!csv) throw std::runtime_error("cannot open real unified result CSV");
		csv << "batch,mode,fragments,merge_logical_rows,dense_logical_rows,"
			"products,output_nnz,packing_ms,static_h2d_ms,route_h2d_ms,"
			"kernel_ms,d2h_ms,verification_ms,verified_rows\n";
		std::vector<RealModeTotal> totals;
		for (RealMode mode : real_requested_modes())
			totals.push_back({mode, real_mode_name(mode),
				std::vector<double>(reps, 0.0)});

		std::unique_ptr<xrt::device> device;
		std::unique_ptr<xrt::kernel> kernel;
		if (!dry_run) {
			if (lower(argv[2]) == "auto" && std::getenv("TARGET_CARD") == nullptr)
				::setenv("TARGET_CARD", "u280", 0);
			device = std::make_unique<xrt::device>(
				host_device::parse_device_arg(argv[2]));
			std::cout << "[HOST] load_xclbin begin: " << argv[1] << std::endl;
			const auto uuid = device->load_xclbin(argv[1]);
			std::cout << "[HOST] load_xclbin complete" << std::endl;
			std::cout << "[HOST] kernel open begin: "
				<< "adaptive_hbm_unified_real_hbm_tapa:{real_1}"
				<< std::endl;
			kernel = std::make_unique<xrt::kernel>(*device, uuid,
				"adaptive_hbm_unified_real_hbm_tapa:"
				"{real_1}",
				xrt::kernel::cu_access_mode::shared);
			std::cout << "[HOST] kernel open complete" << std::endl;
		}

		std::vector<std::pair<id_t, id_t> > pending
			= dim_make_rowmajor_ranges(matrix, index);
		std::reverse(pending.begin(), pending.end());
		unsigned batch_index = 0;
		while (!pending.empty()) {
			const auto range = pending.back();
			pending.pop_back();
			DimSuperbatch packed = dim_pack_rowmajor_range(matrix, index,
				whole_plan, range.first, range.second);
			if (!dim_superbatch_fits(packed)) {
				if (range.second - range.first <= 1)
					throw std::runtime_error("one row exceeds input HBM capacity");
				const id_t middle = range.first + (range.second - range.first) / 2;
				pending.push_back({middle, range.second});
				pending.push_back({range.first, middle});
				continue;
			}
			dim_validate_rpc1_batch(matrix, packed);
			RealFragmentBatch expanded = real_expand_rpc1(&packed);
			for (RealFragmentBatch& batch : real_capacity_slices(expanded)) {
				// Execute an exact half-open physical-fragment range when isolating
				// whether a failure follows one row's data or accumulated protocol
				// state.  This is a diagnostic-only Host slice; the packed fragment
				// ABI presented to the FPGA remains unchanged.
				if (const char* range_text
						= std::getenv("ADAPT_DEBUG_FRAGMENT_RANGE")) {
					const std::string range(range_text);
					const size_t separator = range.find(':');
					if (separator == std::string::npos)
						throw std::runtime_error(
							"ADAPT_DEBUG_FRAGMENT_RANGE must be begin:end");
					const unsigned long begin = std::stoul(range.substr(0, separator));
					const unsigned long end = std::stoul(range.substr(separator + 1));
					if (begin >= end || end > batch.logical_row.size())
						throw std::runtime_error(
							"ADAPT_DEBUG_FRAGMENT_RANGE is outside the batch");
					batch = real_slice(batch, begin, end);
					std::cout << "[HOST] debug fragment range=" << begin
						<< ':' << end << std::endl;
				}
				// Hardware bisection aid: execute a prefix of one already-valid
				// physical-fragment batch, preserving the exact task/geometry ABI.
				// It is ignored unless explicitly set and can never affect the
				// production campaign or its frozen result CSVs.
				if (const char* limit_text = std::getenv("ADAPT_DEBUG_MAX_FRAGMENTS")) {
					const unsigned long requested = std::stoul(limit_text);
					if (requested == 0)
						throw std::runtime_error("ADAPT_DEBUG_MAX_FRAGMENTS must be positive");
					if (requested < batch.logical_row.size()) {
						batch = real_slice(batch, 0, requested);
						std::cout << "[HOST] debug fragment prefix=" << requested
							<< std::endl;
					}
				}
				real_debug_round_robin_tasks(&batch);
				real_debug_limit_tasks_per_shard(&batch);
				real_debug_dump_fragment_tail(batch);
				if (dry_run) {
					std::cout << "real_batch=" << batch_index
						<< " fragments=" << batch.logical_row.size()
						<< " products=" << batch.products
						<< " output_words_upper=" << batch.packet_words_upper
						<< " PACKED\n";
					real_print_dry_run_diagnostics(batch, batch_index);
				} else {
					real_execute_batch(matrix, batch, *device, *kernel, reps,
						batch_index, heavy_threshold, csv, &totals);
				}
				++batch_index;
			}
		}
		if (dry_run) {
			std::cout << "UNIFIED_REAL_HOST_DRY_RUN: PASS batches="
				<< batch_index << " selector_ms=" << selector_ms << '\n';
			return 0;
		}
		for (RealModeTotal& total : totals) {
			for (unsigned rep = 0; rep < reps; ++rep)
				csv << "SAMPLE," << total.name << ",0,0,0,0,0,0,0,0,"
					<< total.kernel_samples_ms[rep] << ",0,0," << rep << '\n';
			csv << "TOTAL," << total.name << ',' << total.fragments << ','
				<< total.merge_logical_rows << ',' << total.dense_logical_rows
				<< ',' << total.products << ',' << total.output_nnz << ",0,"
				<< total.static_h2d_ms << ',' << total.route_h2d_ms << ','
				<< median(total.kernel_samples_ms) << ',' << total.d2h_ms << ','
				<< total.verification_ms << ',' << total.logical_rows << '\n';
			std::cout << "real_whole mode=" << total.name
				<< " kernel_samples_ms=[";
			for (unsigned rep = 0; rep < reps; ++rep) {
				if (rep != 0) std::cout << ',';
				std::cout << total.kernel_samples_ms[rep];
			}
			std::cout << "] median_ms=" << median(total.kernel_samples_ms)
				<< " output_nnz=" << total.output_nnz << " route(M,D)=("
				<< total.merge_logical_rows << ',' << total.dense_logical_rows
				<< ")\n";
		}
		std::cout << "UNIFIED_REAL_HOST_BOARD: PASS batches=" << batch_index
			<< " selector_ms=" << selector_ms
			<< " heavy_threshold=" << heavy_threshold << '\n';
		return 0;
	} catch (const std::exception& error) {
		std::cerr << "error: " << error.what() << '\n';
		return 1;
	}
}
