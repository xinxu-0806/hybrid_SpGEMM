#include "adaptive_hbm_unified_dense_dual_context_tapa.h"

namespace {

constexpr int kBanks = 8;
constexpr int kPhases = 11;
constexpr int kContextColumns = 32768;
constexpr int kAddresses = kContextColumns / kBanks;
constexpr int kLeafWords = kAddresses / 64;
constexpr int kEpochBits = 16;
using Cell = ap_uint<32 + kEpochBits>;

static uint32_t raw_float(float value) {
#pragma HLS INLINE
	union { float f; uint32_t u; } bits;
	bits.f = value;
	return bits.u;
}

static float from_raw(ap_uint<32> value) {
#pragma HLS INLINE
	union { float f; uint32_t u; } bits;
	bits.u = (uint32_t)value;
	return bits.f;
}

static bool nonzero(float value) {
#pragma HLS INLINE
	return (raw_float(value) & 0x7fffffffU) != 0;
}

static ap_uint<4> popcount8(ap_uint<8> value) {
#pragma HLS INLINE
	const ap_uint<3> low = value[0] + value[1] + value[2] + value[3];
	const ap_uint<3> high = value[4] + value[5] + value[6] + value[7];
	return low + high;
}

static float sum11(float value[11]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=value complete
	const float s01 = value[0] + value[1];
	const float s23 = value[2] + value[3];
	const float s45 = value[4] + value[5];
	const float s67 = value[6] + value[7];
	const float s89 = value[8] + value[9];
	const float s03 = s01 + s23;
	const float s47 = s45 + s67;
	const float s810 = s89 + value[10];
	return (s03 + s47) + s810;
}

static DenseDualContextToken end_token() {
#pragma HLS INLINE
	DenseDualContextToken token = 0;
	token[566] = 1;
	return token;
}

// Allocator completion layout shared with the MERGE path:
// context[2:0], generation[10:3], row[42:11].
static ap_uint<64> make_completion(id_t row, ap_uint<3> context,
		ap_uint<8> generation) {
#pragma HLS INLINE
	ap_uint<64> event = 0;
	event.range(2, 0) = context;
	event.range(10, 3) = generation;
	event.range(42, 11) = row;
	return event;
}

static void insert64(ap_uint<512>& word, ap_uint<3> position,
		ap_uint<64> item) {
#pragma HLS INLINE
	switch ((unsigned)position) {
	case 0: word.range(63, 0) = item; break;
	case 1: word.range(127, 64) = item; break;
	case 2: word.range(191, 128) = item; break;
	case 3: word.range(255, 192) = item; break;
	case 4: word.range(319, 256) = item; break;
	case 5: word.range(383, 320) = item; break;
	case 6: word.range(447, 384) = item; break;
	default: word.range(511, 448) = item; break;
	}
}

}  // namespace

void dense_dual_context_read(
		tapa::mmap<const ap_uint<512> > data_in,
		tapa::mmap<const ap_uint<64> > meta_in,
		id_t packet_count,
		tapa::ostream<DenseDualContextToken>& output) {
	for (id_t packet = 0; packet < packet_count; ++packet) {
#pragma HLS PIPELINE II=1
		const ap_uint<64> meta = meta_in[packet];
		DenseDualContextToken token = 0;
		token.range(511, 0) = data_in[packet];
		token.range(543, 512) = meta.range(31, 0);
		token.range(551, 544) = meta.range(39, 32);
		token[552] = meta[53];
		token[553] = meta[51];
		token.range(556, 554) = meta.range(42, 40);
		token.range(564, 557) = meta.range(50, 43);
		token[565] = meta[52];
		token[567] = meta[54];
		output.write(token);
	}
	output.write(end_token());
}

void dense_dual_context_route(
		tapa::istream<DenseDualContextToken>& input,
		tapa::ostream<DenseDualContextToken>& context0,
		tapa::ostream<DenseDualContextToken>& context1) {
	bool done = false;
	while (!done) {
#pragma HLS PIPELINE II=1
		const DenseDualContextToken token = input.read();
		if (token[566]) {
			context0.write(token);
			context1.write(token);
			done = true;
		} else if (token[552]) {
			context1.write(token);
		} else {
			context0.write(token);
		}
	}
}

// Consume exactly one row for one physical context.  This explicit function
// boundary gives HLS a standalone one-packet-per-cycle accumulation loop.
template <int CONTEXT>
static void dense_dual_accumulate_row(
		tapa::istream<DenseDualContextToken>& input,
		id_t column_count,
		ap_uint<kEpochBits> epoch,
		Cell accumulator[kBanks][kPhases][kAddresses],
		ap_uint<64> leaf_bits[kBanks][kPhases][kLeafWords],
		ap_uint<kEpochBits> leaf_tag[kBanks][kPhases][kLeafWords],
		id_t& active_row,
		ap_uint<3>& active_context,
		ap_uint<8>& active_generation,
		bool& active_wide,
		bool& terminal,
		id_t& input_packets,
		id_t& protocol_errors) {
#pragma HLS INLINE off
#pragma HLS DEPENDENCE variable=accumulator inter false
#pragma HLS DEPENDENCE variable=leaf_bits inter false
#pragma HLS DEPENDENCE variable=leaf_tag inter false
	bool active = false;
	bool row_done = false;
	ap_uint<4> phase = 0;
	active_row = 0;
	active_context = 0;
	active_generation = 0;
	active_wide = false;
	terminal = false;
	while (!row_done && !terminal) {
#pragma HLS PIPELINE II=1
		const DenseDualContextToken token = input.read();
		if (token[566]) {
			if (active) ++protocol_errors;
			terminal = true;
			continue;
		}

		const id_t row = token.range(543, 512);
		const ap_uint<1> context_error
			= token[552] != (ap_uint<1>)CONTEXT;
		const ap_uint<3> allocator_context = token.range(556, 554);
		const ap_uint<8> generation = token.range(564, 557);
		const bool wide = token[567];
		if (token[553]) {
			// A row may be empty after packet-local FP32 cancellation.  In that
			// case EOR is its first accumulator-visible token and is a valid
			// completion, not a protocol error.
			const ap_uint<1> eor_error = (active && (row != active_row
				|| allocator_context != active_context
				|| generation != active_generation || wide != active_wide))
				|| token.range(551, 544) != 0 || token[565];
			protocol_errors += (ap_uint<2>)context_error + eor_error;
			if (!active) {
				active_row = row;
				active_context = allocator_context;
				active_generation = generation;
				active_wide = wide;
			}
			row_done = true;
			continue;
		}

		ap_uint<1> row_error = 0;
		if (!active) {
			active = true;
			active_row = row;
			active_context = allocator_context;
			active_generation = generation;
			active_wide = wide;
		} else if (row != active_row || allocator_context != active_context
				|| generation != active_generation || wide != active_wide) {
			row_error = 1;
		}
		++input_packets;
		const ap_uint<8> mask = token.range(551, 544);
		ap_uint<8> duplicate_mask = 0;
		ap_uint<8> range_error_mask = 0;
		for (int bank = 0; bank < kBanks; ++bank) {
#pragma HLS UNROLL
			ap_uint<8> match = 0;
			ap_uint<64> lane_item[8];
#pragma HLS ARRAY_PARTITION variable=lane_item complete
			for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
				lane_item[lane] = token.range(lane * 64 + 63, lane * 64);
				match[lane] = mask[lane]
					&& lane_item[lane].range(2, 0) == (ap_uint<3>)bank;
			}
			const ap_uint<8> onehot = match & (ap_uint<8>)(~match + 1);
			duplicate_mask[bank] = (match & (match - 1)) != 0;
			if (match != 0) {
				const ap_uint<64> item = onehot[0] ? lane_item[0]
					: onehot[1] ? lane_item[1] : onehot[2] ? lane_item[2]
					: onehot[3] ? lane_item[3] : onehot[4] ? lane_item[4]
					: onehot[5] ? lane_item[5] : onehot[6] ? lane_item[6]
					: lane_item[7];
				const id_t col = item.range(31, 0);
				if (col >= column_count || col >= kContextColumns) {
					range_error_mask[bank] = 1;
				} else {
					const ap_uint<12> address = col >> 3;
					const Cell cell = accumulator[bank][phase][address];
					const bool seen
						= cell.range(kEpochBits + 31, 32) == epoch;
					const float previous = seen
						? from_raw(cell.range(31, 0)) : 0.0f;
					const float updated = previous
						+ from_raw(item.range(63, 32));
#pragma HLS BIND_OP variable=updated op=fadd impl=fulldsp latency=5
					Cell next = 0;
					next.range(31, 0) = raw_float(updated);
					next.range(kEpochBits + 31, 32) = epoch;
					accumulator[bank][phase][address] = next;
					const ap_uint<6> leaf_word = address >> 6;
					const ap_uint<6> leaf_bit = address.range(5, 0);
					ap_uint<64> bits
						= leaf_tag[bank][phase][leaf_word] == epoch
						? leaf_bits[bank][phase][leaf_word] : (ap_uint<64>)0;
					bits[leaf_bit] = 1;
					leaf_bits[bank][phase][leaf_word] = bits;
					leaf_tag[bank][phase][leaf_word] = epoch;
				}
			}
		}
		const ap_uint<2> header_errors = (ap_uint<2>)context_error
			+ row_error + (ap_uint<1>)token[565];
		const ap_uint<5> bank_errors
			= (ap_uint<5>)popcount8(duplicate_mask)
			+ (ap_uint<5>)popcount8(range_error_mask);
		protocol_errors += header_errors + bank_errors;
		phase = phase == kPhases - 1
			? (ap_uint<4>)0 : (ap_uint<4>)(phase + 1);
	}
}

// Emit one completed row.  The sibling physical context is a separate TAPA
// task, so it can accumulate its next row while this scan is running.
template <int CONTEXT>
static void dense_dual_extract_row(
		id_t active_row,
		ap_uint<3> active_context,
		ap_uint<8> active_generation,
		bool active_wide,
		id_t column_count,
		ap_uint<kEpochBits> epoch,
		Cell accumulator[kBanks][kPhases][kAddresses],
		ap_uint<64> leaf_bits[kBanks][kPhases][kLeafWords],
		ap_uint<kEpochBits> leaf_tag[kBanks][kPhases][kLeafWords],
		tapa::ostream<DenseDualContextToken>& output,
		id_t& extract_cycles,
		id_t& output_packets,
		id_t& output_items) {
#pragma HLS INLINE off
	for (int word_index = 0; word_index < kLeafWords; ++word_index) {
		ap_uint<64> candidate[kBanks];
#pragma HLS ARRAY_PARTITION variable=candidate complete
		for (int bank = 0; bank < kBanks; ++bank) {
#pragma HLS UNROLL
			ap_uint<64> combined = 0;
			for (int p = 0; p < kPhases; ++p) {
#pragma HLS UNROLL
				if (leaf_tag[bank][p][word_index] == epoch)
					combined |= leaf_bits[bank][p][word_index];
			}
			candidate[bank] = combined;
		}
		for (int bit = 0; bit < 64; ++bit) {
#pragma HLS PIPELINE II=1
			ap_uint<512> packed = 0;
			ap_uint<8> valid_mask = 0;
			const int address = word_index * 64 + bit;
			for (int bank = 0; bank < kBanks; ++bank) {
#pragma HLS UNROLL
				if (candidate[bank][bit]) {
					float value[kPhases];
#pragma HLS ARRAY_PARTITION variable=value complete
					for (int p = 0; p < kPhases; ++p) {
#pragma HLS UNROLL
						const Cell cell = accumulator[bank][p][address];
						value[p] = cell.range(kEpochBits + 31, 32) == epoch
							? from_raw(cell.range(31, 0)) : 0.0f;
					}
					const float sum = sum11(value);
					if (nonzero(sum)) {
						const id_t col = (address << 3) | bank;
						if (col < column_count) {
							ap_uint<64> item = 0;
							item.range(31, 0) = col;
							item.range(63, 32) = raw_float(sum);
							insert64(packed, bank, item);
							valid_mask[bank] = 1;
						}
					}
				}
			}
			if (valid_mask != 0) {
				DenseDualContextToken result = 0;
				result.range(511, 0) = packed;
				result.range(543, 512) = active_row;
				result.range(551, 544) = valid_mask;
				result[552] = CONTEXT;
				result.range(556, 554) = active_context;
				result.range(564, 557) = active_generation;
				result[567] = active_wide;
				output.write(result);
				++output_packets;
				output_items += popcount8(valid_mask);
			}
			++extract_cycles;
		}
	}
	DenseDualContextToken eor = 0;
	eor.range(543, 512) = active_row;
	eor[552] = CONTEXT;
	eor[553] = 1;
	eor.range(556, 554) = active_context;
	eor.range(564, 557) = active_generation;
	eor[567] = active_wide;
	output.write(eor);
}

template <int CONTEXT>
static void dense_dual_context_engine_impl(
		tapa::istream<DenseDualContextToken>& input,
		id_t column_count,
		tapa::ostream<DenseDualContextToken>& output,
		tapa::ostream<ap_uint<192> >& statistics_output) {
	static Cell accumulator[kBanks][kPhases][kAddresses];
	static ap_uint<64> leaf_bits[kBanks][kPhases][kLeafWords];
	static ap_uint<kEpochBits> leaf_tag[kBanks][kPhases][kLeafWords];
	static bool initialized = false;
	static ap_uint<kEpochBits> epoch_state = 0;
#pragma HLS ARRAY_PARTITION variable=accumulator complete dim=1
#pragma HLS ARRAY_PARTITION variable=accumulator complete dim=2
#pragma HLS ARRAY_PARTITION variable=leaf_bits complete dim=1
#pragma HLS ARRAY_PARTITION variable=leaf_bits complete dim=2
#pragma HLS ARRAY_PARTITION variable=leaf_tag complete dim=1
#pragma HLS ARRAY_PARTITION variable=leaf_tag complete dim=2
#pragma HLS BIND_STORAGE variable=accumulator type=ram_t2p impl=uram latency=2
#pragma HLS BIND_STORAGE variable=leaf_bits type=ram_t2p impl=bram latency=2
#pragma HLS BIND_STORAGE variable=leaf_tag type=ram_t2p impl=bram latency=2
#pragma HLS DEPENDENCE variable=accumulator inter false
#pragma HLS DEPENDENCE variable=leaf_bits inter false
#pragma HLS DEPENDENCE variable=leaf_tag inter false

	if (!initialized) {
		for (int address = 0; address < kAddresses; ++address) {
#pragma HLS PIPELINE II=1
			for (int bank = 0; bank < kBanks; ++bank) {
#pragma HLS UNROLL
				for (int phase = 0; phase < kPhases; ++phase) {
#pragma HLS UNROLL
					accumulator[bank][phase][address] = 0;
					if (address < kLeafWords) {
						leaf_bits[bank][phase][address] = 0;
						leaf_tag[bank][phase][address] = 0;
					}
				}
			}
		}
		initialized = true;
	}

	id_t rows = 0;
	id_t input_packets = 0;
	id_t extract_cycles = 0;
	id_t output_packets = 0;
	id_t output_items = 0;
	id_t protocol_errors = 0;
	ap_uint<kEpochBits> epoch = epoch_state + 1;
	epoch_state = epoch;
	bool done = false;
	while (!done) {
		id_t active_row = 0;
		ap_uint<3> active_context = 0;
		ap_uint<8> active_generation = 0;
		bool active_wide = false;
		bool terminal = false;
		dense_dual_accumulate_row<CONTEXT>(input, column_count, epoch,
			accumulator, leaf_bits, leaf_tag, active_row, active_context,
			active_generation, active_wide, terminal,
			input_packets, protocol_errors);
		if (terminal) {
			output.write(end_token());
			done = true;
		} else {
			dense_dual_extract_row<CONTEXT>(active_row, active_context,
				active_generation, active_wide, column_count, epoch,
				accumulator, leaf_bits, leaf_tag, output, extract_cycles,
				output_packets, output_items);
			++rows;
			epoch = epoch + 1;
			epoch_state = epoch;
		}
	}
	ap_uint<192> statistics = 0;
	statistics.range(31, 0) = rows;
	statistics.range(63, 32) = input_packets;
	statistics.range(95, 64) = extract_cycles;
	statistics.range(127, 96) = output_packets;
	statistics.range(159, 128) = output_items;
	statistics.range(191, 160) = protocol_errors;
	statistics_output.write(statistics);

#if 0  // Superseded monolithic reference: kept temporarily for report diffing.
	id_t rows = 0;
	id_t input_packets = 0;
	id_t extract_cycles = 0;
	id_t output_packets = 0;
	id_t output_items = 0;
	id_t protocol_errors = 0;
	bool active = false;
	id_t active_row = 0;
	ap_uint<4> phase = 0;
	ap_uint<kEpochBits> epoch = epoch_state + 1;
	epoch_state = epoch;
	bool done = false;
	while (!done) {
		const DenseDualContextToken token = input.read();
		if (token[554]) {
			if (active) ++protocol_errors;
			output.write(token);
			done = true;
			continue;
		}
		const id_t row = token.range(543, 512);
		if (token[552] != (ap_uint<1>)CONTEXT) ++protocol_errors;
		if (token[553]) {
			if (!active || row != active_row || token.range(551, 544) != 0)
				++protocol_errors;
			// A context-local extractor can run while the other physical engine
			// keeps accumulating.  Eight banks are emitted together for one
			// address group, giving one 512-bit DENSE extraction stream.
			for (int word_index = 0; word_index < kLeafWords; ++word_index) {
				ap_uint<64> candidate[kBanks];
#pragma HLS ARRAY_PARTITION variable=candidate complete
				for (int bank = 0; bank < kBanks; ++bank) {
#pragma HLS UNROLL
					ap_uint<64> combined = 0;
					for (int p = 0; p < kPhases; ++p) {
#pragma HLS UNROLL
						if (leaf_tag[bank][p][word_index] == epoch)
							combined |= leaf_bits[bank][p][word_index];
					}
					candidate[bank] = combined;
				}
				for (int bit = 0; bit < 64; ++bit) {
#pragma HLS PIPELINE II=1
					ap_uint<512> packed = 0;
					ap_uint<8> valid_mask = 0;
					const int address = word_index * 64 + bit;
					for (int bank = 0; bank < kBanks; ++bank) {
#pragma HLS UNROLL
						if (candidate[bank][bit]) {
							float value[kPhases];
#pragma HLS ARRAY_PARTITION variable=value complete
							for (int p = 0; p < kPhases; ++p) {
#pragma HLS UNROLL
								const Cell cell = accumulator[bank][p][address];
								value[p] = cell.range(kEpochBits + 31, 32) == epoch
									? from_raw(cell.range(31, 0)) : 0.0f;
							}
							const float sum = sum11(value);
							if (nonzero(sum)) {
								const id_t col = (address << 3) | bank;
								if (col < column_count) {
									ap_uint<64> item = 0;
									item.range(31, 0) = col;
									item.range(63, 32) = raw_float(sum);
									insert64(packed, bank, item);
									valid_mask[bank] = 1;
								}
							}
						}
					}
					if (valid_mask != 0) {
						DenseDualContextToken result = 0;
						result.range(511, 0) = packed;
						result.range(543, 512) = active_row;
						result.range(551, 544) = valid_mask;
						result[552] = CONTEXT;
						output.write(result);
						++output_packets;
						output_items += popcount8(valid_mask);
					}
					++extract_cycles;
				}
			}
			DenseDualContextToken eor = 0;
			eor.range(543, 512) = active_row;
			eor[552] = CONTEXT;
			eor[553] = 1;
			output.write(eor);
			++rows;
			active = false;
			phase = 0;
			epoch = epoch + 1;
			epoch_state = epoch;
			continue;
		}

		if (!active) { active = true; active_row = row; }
		else if (row != active_row) ++protocol_errors;
		++input_packets;
		const ap_uint<8> mask = token.range(551, 544);
		for (int bank = 0; bank < kBanks; ++bank) {
#pragma HLS UNROLL
			ap_uint<8> match = 0;
			ap_uint<64> lane_item[8];
#pragma HLS ARRAY_PARTITION variable=lane_item complete
			for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
				lane_item[lane] = token.range(lane * 64 + 63, lane * 64);
				match[lane] = mask[lane]
					&& lane_item[lane].range(2, 0) == (ap_uint<3>)bank;
			}
			const ap_uint<8> onehot = match & (ap_uint<8>)(~match + 1);
			if ((match & (match - 1)) != 0) ++protocol_errors;
			if (match != 0) {
				const ap_uint<64> item = onehot[0] ? lane_item[0]
					: onehot[1] ? lane_item[1] : onehot[2] ? lane_item[2]
					: onehot[3] ? lane_item[3] : onehot[4] ? lane_item[4]
					: onehot[5] ? lane_item[5] : onehot[6] ? lane_item[6]
					: lane_item[7];
				const id_t col = item.range(31, 0);
				if (col >= column_count || col >= kContextColumns) {
					++protocol_errors;
				} else {
					const ap_uint<12> address = col >> 3;
					const Cell cell = accumulator[bank][phase][address];
					const bool seen
						= cell.range(kEpochBits + 31, 32) == epoch;
					const float previous = seen
						? from_raw(cell.range(31, 0)) : 0.0f;
					const float updated = previous
						+ from_raw(item.range(63, 32));
#pragma HLS BIND_OP variable=updated op=fadd impl=fulldsp latency=5
					Cell next = 0;
					next.range(31, 0) = raw_float(updated);
					next.range(kEpochBits + 31, 32) = epoch;
					accumulator[bank][phase][address] = next;
					const ap_uint<6> leaf_word = address >> 6;
					const ap_uint<6> leaf_bit = address.range(5, 0);
					ap_uint<64> bits = leaf_tag[bank][phase][leaf_word] == epoch
						? leaf_bits[bank][phase][leaf_word] : (ap_uint<64>)0;
					bits[leaf_bit] = 1;
					leaf_bits[bank][phase][leaf_word] = bits;
					leaf_tag[bank][phase][leaf_word] = epoch;
				}
			}
		}
		phase = phase == kPhases - 1
			? (ap_uint<4>)0 : (ap_uint<4>)(phase + 1);
	}
	ap_uint<192> statistics = 0;
	statistics.range(31, 0) = rows;
	statistics.range(63, 32) = input_packets;
	statistics.range(95, 64) = extract_cycles;
	statistics.range(127, 96) = output_packets;
	statistics.range(159, 128) = output_items;
	statistics.range(191, 160) = protocol_errors;
	statistics_output.write(statistics);
#endif
}

void dense_dual_context_engine0(
		tapa::istream<DenseDualContextToken>& input, id_t column_count,
		tapa::ostream<DenseDualContextToken>& output,
		tapa::ostream<ap_uint<192> >& statistics_output) {
	dense_dual_context_engine_impl<0>(input, column_count, output,
		statistics_output);
}

void dense_dual_context_engine1(
		tapa::istream<DenseDualContextToken>& input, id_t column_count,
		tapa::ostream<DenseDualContextToken>& output,
		tapa::ostream<ap_uint<192> >& statistics_output) {
	dense_dual_context_engine_impl<1>(input, column_count, output,
		statistics_output);
}

void dense_dual_context_write(
		tapa::istream<DenseDualContextToken>& input,
		tapa::istream<ap_uint<192> >& statistics_input,
		tapa::mmap<ap_uint<512> > data_out,
		tapa::mmap<ap_uint<64> > meta_out,
		tapa::mmap<ap_uint<64> > completion_out,
		tapa::mmap<id_t> stats_out) {
	id_t cursor = 0;
	id_t completion_cursor = 0;
	bool done = false;
	while (!done) {
#pragma HLS PIPELINE II=1
		const DenseDualContextToken token = input.read();
		if (token[566]) { done = true; continue; }
		data_out[cursor] = token.range(511, 0);
		ap_uint<64> meta = 0;
		meta.range(31, 0) = token.range(543, 512);
		meta.range(39, 32) = token.range(551, 544);
		meta.range(42, 40) = token.range(556, 554);
		meta.range(50, 43) = token.range(564, 557);
		meta[51] = token[553];
		meta[52] = token[565];
		meta[53] = token[552];
		meta_out[cursor++] = meta;
		// The allocator may reuse this logical context only after extraction
		// has emitted its EOR into the output path.
		if (token[553]) {
			completion_out[completion_cursor++] = make_completion(
				token.range(543, 512), token.range(556, 554),
				token.range(564, 557));
		}
	}
	const ap_uint<192> statistics = statistics_input.read();
	for (int index = 0; index < 6; ++index) {
#pragma HLS PIPELINE II=1
		stats_out[index] = statistics.range(index * 32 + 31, index * 32);
	}
}

void adaptive_hbm_unified_dense_dual_context_tapa(
		tapa::mmap<const ap_uint<512> > data_in,
		tapa::mmap<const ap_uint<64> > meta_in,
		id_t packet_count,
		id_t column_count,
		tapa::mmap<ap_uint<512> > output0,
		tapa::mmap<ap_uint<64> > output_meta0,
		tapa::mmap<ap_uint<64> > output_completion0,
		tapa::mmap<id_t> statistics0,
		tapa::mmap<ap_uint<512> > output1,
		tapa::mmap<ap_uint<64> > output_meta1,
		tapa::mmap<ap_uint<64> > output_completion1,
		tapa::mmap<id_t> statistics1) {
	tapa::stream<DenseDualContextToken, 16> raw("raw");
	tapa::streams<DenseDualContextToken, 2, 16> context_input("context_input");
	tapa::streams<DenseDualContextToken, 2, 16> context_output("context_output");
	tapa::streams<ap_uint<192>, 2, 2> context_statistics("context_statistics");
	tapa::task()
		.invoke(dense_dual_context_read, data_in, meta_in, packet_count, raw)
		.invoke(dense_dual_context_route, raw, context_input[0], context_input[1])
		.invoke(dense_dual_context_engine0, context_input[0], column_count,
			context_output[0], context_statistics[0])
		.invoke(dense_dual_context_engine1, context_input[1], column_count,
			context_output[1], context_statistics[1])
		.invoke(dense_dual_context_write, context_output[0], context_statistics[0],
			output0, output_meta0, output_completion0, statistics0)
		.invoke(dense_dual_context_write, context_output[1], context_statistics[1],
			output1, output_meta1, output_completion1, statistics1);
}
