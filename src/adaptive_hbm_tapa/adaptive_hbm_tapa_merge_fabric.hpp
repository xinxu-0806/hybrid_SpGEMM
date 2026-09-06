// Internal persistent MERGE fabric for adaptive_hbm_tapa_scalable_windowed.cpp.
//
// This file is intentionally included after the unified packet types and FP32
// helpers have been declared.  It is not a third Host-visible algorithm.  It
// replaces the former single global min-column selector inside MERGE/DIRECT:
// ordinary rows occupy one of eight row-context roots, while a heavy row is
// hash-partitioned over all eight roots.  Every numerical stream carries an
// explicit row tag and every row ends with an EOR token containing the source
// product count and protocol status.

struct TapaBankedMergeItem {
	id_t row;
	ap_uint<16> col;
	ap_uint<32> value;
	id_t products;
	ap_uint<2> errors;
	ap_uint<3> destination;
	bool eor;
	bool heavy;
	bool broadcast;
	bool last;
};

using TapaBankedMergeStream = hls::stream<TapaBankedMergeItem>;

struct TapaMergeSparsePacket {
	id_t row;
	ap_uint<512> items;
	ap_uint<8> mask;
	id_t products;
	ap_uint<2> errors;
	bool eor;
	bool heavy;
	bool last;
};

struct TapaMergeCompactPacket {
	id_t row;
	ap_uint<512> items;
	ap_uint<4> count;
	id_t products;
	ap_uint<2> errors;
	bool eor;
	bool heavy;
	bool last;
};

using TapaMergeSparseStream = hls::stream<TapaMergeSparsePacket>;
using TapaMergeCompactStream = hls::stream<TapaMergeCompactPacket>;

static ap_uint<3> tapa_merge_hash(ap_uint<16> col) {
#pragma HLS INLINE
	return (col ^ (col >> 3) ^ (col >> 6)) & 7;
}

static bool tapa_merge_same_key(const TapaBankedMergeItem& lhs,
		const TapaBankedMergeItem& rhs) {
#pragma HLS INLINE
	return lhs.row == rhs.row && lhs.eor == rhs.eor
		&& lhs.col == rhs.col && lhs.destination == rhs.destination;
}

static ap_uint<52> tapa_merge_item_key(const TapaBankedMergeItem& item) {
#pragma HLS INLINE
	return ((ap_uint<52>)item.row << 20)
		| ((ap_uint<20>)item.eor << 19)
		| ((ap_uint<19>)item.col << 3)
		| item.destination;
}

static TapaBankedMergeItem tapa_merge_terminal() {
#pragma HLS INLINE
	TapaBankedMergeItem item;
	item.row = 0;
	item.col = 0;
	item.value = 0;
	item.products = 0;
	item.errors = 0;
	item.destination = 0;
	item.eor = false;
	item.heavy = false;
	item.broadcast = false;
	item.last = true;
	return item;
}

void tapa_merge_route_source(
		tapa::istream<TapaProductPacket>& input,
		tapa::ostream<TapaProductPacket>& bypass,
		tapa::ostream<TapaBankedMergeItem>& merge) {
#pragma HLS INLINE off
	bool done = false;
	while (!done) {
		const TapaProductPacket packet = input.read();
		if (packet[552]) {
			bypass.write(packet);
			done = true;
			break;
		}
		const ap_uint<2> mode = packet.range(549, 548);
		const bool merge_route = mode == ADAPT_FP32_MERGE
			|| mode == ADAPT_FP32_DIRECT;
		if (!merge_route) {
			bypass.write(packet);
			continue;
		}

		const id_t row = packet.range(547, 516);
		const bool heavy = packet[554];
		if (packet[551]) {
			TapaBankedMergeItem eor;
			eor.row = row;
			eor.col = 0;
			eor.value = 0;
			eor.products = packet.range(31, 0);
			eor.errors = packet[553] ? ap_uint<2>(1) : ap_uint<2>(0);
			eor.destination = row & 7;
			eor.eor = true;
			eor.heavy = heavy;
			// A boundary is also the empty-branch marker required by every
			// merge/forward node.  It is duplicated one destination bit at a
			// time; ordinary padding boundaries are removed at the leaves.
			eor.broadcast = true;
			eor.last = false;
			merge.write(eor);
			continue;
		}

		const ap_uint<4> valid = packet.range(515, 512);
		for (int lane = 0; lane < 8; ++lane) {
#pragma HLS PIPELINE II=1
			if (lane >= valid) continue;
			TapaBankedMergeItem item;
			item.row = row;
			item.col = packet.range(lane * 64 + 15, lane * 64);
			item.value = packet.range(lane * 64 + 63, lane * 64 + 32);
			item.products = 0;
			item.errors = valid == 0 || valid > 8 ? ap_uint<2>(1)
				: ap_uint<2>(0);
			item.destination = heavy
				? tapa_merge_hash(item.col) : (ap_uint<3>)(row & 7);
			item.eor = false;
			item.heavy = heavy;
			item.broadcast = false;
			item.last = false;
			merge.write(item);
		}
	}
	merge.write(tapa_merge_terminal());
}

// Split one destination bit.  Numerical tokens take one branch; EOR control
// tokens take both so every downstream merge can close an empty row without
// waiting for the end of the complete matrix stream.
template <int BIT>
static void tapa_merge_flexible_split_impl(
		tapa::istream<TapaBankedMergeItem>& input,
		tapa::ostream<TapaBankedMergeItem>& low,
		tapa::ostream<TapaBankedMergeItem>& high) {
#pragma HLS INLINE off
	bool done = false;
	while (!done) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=1048576 avg=4096
		const TapaBankedMergeItem source = input.read();
		if (source.last) {
			const TapaBankedMergeItem terminal = tapa_merge_terminal();
			low.write(terminal);
			high.write(terminal);
			done = true;
		} else if (source.broadcast) {
			TapaBankedMergeItem item0 = source;
			TapaBankedMergeItem item1 = source;
			item0.destination[BIT] = 0;
			item1.destination[BIT] = 1;
			if (BIT == 2) {
				item0.broadcast = false;
				item1.broadcast = false;
			}
			low.write(item0);
			high.write(item1);
		} else if (source.destination[BIT]) {
			high.write(source);
		} else {
			low.write(source);
		}
	}
}

void tapa_merge_flexible_split0(
		tapa::istream<TapaBankedMergeItem>& input,
		tapa::ostream<TapaBankedMergeItem>& low,
		tapa::ostream<TapaBankedMergeItem>& high) {
	tapa_merge_flexible_split_impl<0>(input, low, high);
}

void tapa_merge_flexible_split1(
		tapa::istream<TapaBankedMergeItem>& input,
		tapa::ostream<TapaBankedMergeItem>& low,
		tapa::ostream<TapaBankedMergeItem>& high) {
	tapa_merge_flexible_split_impl<1>(input, low, high);
}

void tapa_merge_flexible_split2(
		tapa::istream<TapaBankedMergeItem>& input,
		tapa::ostream<TapaBankedMergeItem>& low,
		tapa::ostream<TapaBankedMergeItem>& high) {
	tapa_merge_flexible_split_impl<2>(input, low, high);
}

// One timing-safe eight-way root.  TAPA replicates this task eight times, one
// per row-context/hash bank.  Keeping each root as its own task avoids the
// Vitis HLS 2022.2 Block_entry scheduler crash triggered by a single wrapper
// containing all 56 binary nodes, while retaining eight concurrently active
// physical roots and a balanced FP32 reduction tree.
void tapa_merge_reduce8(
		tapa::istream<TapaBankedMergeItem>& input0,
		tapa::istream<TapaBankedMergeItem>& input1,
		tapa::istream<TapaBankedMergeItem>& input2,
		tapa::istream<TapaBankedMergeItem>& input3,
		tapa::istream<TapaBankedMergeItem>& input4,
		tapa::istream<TapaBankedMergeItem>& input5,
		tapa::istream<TapaBankedMergeItem>& input6,
		tapa::istream<TapaBankedMergeItem>& input7,
		tapa::ostream<TapaBankedMergeItem>& output) {
#pragma HLS INLINE off
	TapaBankedMergeItem head[8];
	bool valid[8];
	bool ended[8];
#pragma HLS ARRAY_PARTITION variable=head complete
#pragma HLS ARRAY_PARTITION variable=valid complete
#pragma HLS ARRAY_PARTITION variable=ended complete
	for (int source = 0; source < 8; ++source) {
#pragma HLS UNROLL
		valid[source] = false;
		ended[source] = false;
	}
	while (!(ended[0] && ended[1] && ended[2] && ended[3]
			&& ended[4] && ended[5] && ended[6] && ended[7])) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=1048576 avg=4096
#define TAPA_ROOT_REFILL(S, STREAM) \
		if (!valid[S] && !ended[S]) { \
			const TapaBankedMergeItem item = STREAM.read(); \
			if (item.last) ended[S] = true; \
			else { head[S] = item; valid[S] = true; } \
		}
		TAPA_ROOT_REFILL(0, input0)
		TAPA_ROOT_REFILL(1, input1)
		TAPA_ROOT_REFILL(2, input2)
		TAPA_ROOT_REFILL(3, input3)
		TAPA_ROOT_REFILL(4, input4)
		TAPA_ROOT_REFILL(5, input5)
		TAPA_ROOT_REFILL(6, input6)
		TAPA_ROOT_REFILL(7, input7)
#undef TAPA_ROOT_REFILL
		const bool ready = (valid[0] || ended[0])
			&& (valid[1] || ended[1]) && (valid[2] || ended[2])
			&& (valid[3] || ended[3]) && (valid[4] || ended[4])
			&& (valid[5] || ended[5]) && (valid[6] || ended[6])
			&& (valid[7] || ended[7]);
		const bool active = valid[0] || valid[1] || valid[2] || valid[3]
			|| valid[4] || valid[5] || valid[6] || valid[7];
		if (!ready || !active) continue;
		ap_uint<52> key[8];
#pragma HLS ARRAY_PARTITION variable=key complete
		for (int source = 0; source < 8; ++source) {
#pragma HLS UNROLL
			key[source] = valid[source]
				? tapa_merge_item_key(head[source]) : ~ap_uint<52>(0);
		}
		const ap_uint<52> min01 = key[0] < key[1] ? key[0] : key[1];
		const ap_uint<52> min23 = key[2] < key[3] ? key[2] : key[3];
		const ap_uint<52> min45 = key[4] < key[5] ? key[4] : key[5];
		const ap_uint<52> min67 = key[6] < key[7] ? key[6] : key[7];
		const ap_uint<52> min03 = min01 < min23 ? min01 : min23;
		const ap_uint<52> min47 = min45 < min67 ? min45 : min67;
		const ap_uint<52> minimum_key = min03 < min47 ? min03 : min47;
		TapaBankedMergeItem minimum = key[0] == minimum_key ? head[0]
			: key[1] == minimum_key ? head[1]
			: key[2] == minimum_key ? head[2]
			: key[3] == minimum_key ? head[3]
			: key[4] == minimum_key ? head[4]
			: key[5] == minimum_key ? head[5]
			: key[6] == minimum_key ? head[6] : head[7];
		bool consume[8];
		float contribution[8];
	id_t products[8];
	ap_uint<2> errors[8];
#pragma HLS ARRAY_PARTITION variable=consume complete
#pragma HLS ARRAY_PARTITION variable=contribution complete
#pragma HLS ARRAY_PARTITION variable=products complete
#pragma HLS ARRAY_PARTITION variable=errors complete
		for (int source = 0; source < 8; ++source) {
#pragma HLS UNROLL
			consume[source] = valid[source] && key[source] == minimum_key
				&& tapa_merge_same_key(head[source], minimum);
			contribution[source] = consume[source] && !head[source].eor
				? tapa_bits_to_float(head[source].value) : 0.0f;
			products[source] = consume[source] ? head[source].products : 0;
			errors[source] = consume[source]
				? head[source].errors : ap_uint<2>(0);
			if (consume[source]) valid[source] = false;
		}
		const float sum = tapa_sum8_fp32(contribution[0], contribution[1],
			contribution[2], contribution[3], contribution[4], contribution[5],
			contribution[6], contribution[7]);
		minimum.products = ((products[0] + products[1])
			+ (products[2] + products[3])) + ((products[4] + products[5])
			+ (products[6] + products[7]));
		minimum.errors = (errors[0] | errors[1] | errors[2] | errors[3])
			| (errors[4] | errors[5] | errors[6] | errors[7]);
		minimum.last = false;
		if (minimum.eor) {
			minimum.value = 0;
			output.write(minimum);
		} else if (tapa_nonzero(sum)) {
			minimum.value = tapa_float_to_bits(sum);
			output.write(minimum);
		}
	}
	output.write(tapa_merge_terminal());
}

void tapa_merge_reduce2(tapa::istream<TapaBankedMergeItem>& input0,
		tapa::istream<TapaBankedMergeItem>& input1,
		tapa::ostream<TapaBankedMergeItem>& output) {
#pragma HLS INLINE off
	TapaBankedMergeItem head[2];
	bool valid[2] = {false, false};
	bool ended[2] = {false, false};
#pragma HLS ARRAY_PARTITION variable=head complete
#pragma HLS ARRAY_PARTITION variable=valid complete
#pragma HLS ARRAY_PARTITION variable=ended complete
	while (!(ended[0] && ended[1])) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=1048576 avg=4096
		if (!valid[0] && !ended[0]) {
			const TapaBankedMergeItem item = input0.read();
			if (item.last) ended[0] = true;
			else { head[0] = item; valid[0] = true; }
		}
		if (!valid[1] && !ended[1]) {
			const TapaBankedMergeItem item = input1.read();
			if (item.last) ended[1] = true;
			else { head[1] = item; valid[1] = true; }
		}
		const bool ready = (valid[0] || ended[0]) && (valid[1] || ended[1]);
		if (!ready || (!valid[0] && !valid[1])) continue;
		const ap_uint<52> key0 = valid[0]
			? tapa_merge_item_key(head[0]) : ~ap_uint<52>(0);
		const ap_uint<52> key1 = valid[1]
			? tapa_merge_item_key(head[1]) : ~ap_uint<52>(0);
		const ap_uint<52> minimum_key = key0 < key1 ? key0 : key1;
		const TapaBankedMergeItem minimum
			= key0 == minimum_key ? head[0] : head[1];
		const bool consume0 = valid[0] && key0 == minimum_key
			&& tapa_merge_same_key(head[0], minimum);
		const bool consume1 = valid[1] && key1 == minimum_key
			&& tapa_merge_same_key(head[1], minimum);
		const float value0 = consume0 && !head[0].eor
			? tapa_bits_to_float(head[0].value) : 0.0f;
		const float value1 = consume1 && !head[1].eor
			? tapa_bits_to_float(head[1].value) : 0.0f;
		const float sum = value0 + value1;
#pragma HLS BIND_OP variable=sum op=fadd impl=fulldsp latency=5
		if (consume0) valid[0] = false;
		if (consume1) valid[1] = false;
		TapaBankedMergeItem result = minimum;
		result.products = (consume0 ? head[0].products : (id_t)0)
			+ (consume1 ? head[1].products : (id_t)0);
		result.errors = (consume0 ? head[0].errors : ap_uint<2>(0))
			| (consume1 ? head[1].errors : ap_uint<2>(0));
		result.last = false;
		if (minimum.eor) {
			result.value = 0;
			output.write(result);
		} else if (tapa_nonzero(sum)) {
			result.value = tapa_float_to_bits(sum);
			output.write(result);
		}
	}
	output.write(tapa_merge_terminal());
}

// Ordinary-row EORs were broadcast only as internal empty-branch markers.
// Keep the marker at its owning row-context leaf and remove the other seven;
// heavy rows genuinely occupy all hash leaves and retain all eight EORs.
void tapa_merge_flexible_filter_padding(
		tapa::istream<TapaBankedMergeItem>& input,
		tapa::ostream<TapaBankedMergeItem>& output) {
#pragma HLS INLINE off
	bool done = false;
	while (!done) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=1048576 avg=4096
		const TapaBankedMergeItem item = input.read();
		if (item.last) {
			output.write(item);
			done = true;
		} else if (!(item.eor && !item.heavy
				&& item.destination != (ap_uint<3>)(item.row & 7))) {
			output.write(item);
		}
	}
}

static ap_uint<512> tapa_merge_insert64(ap_uint<512> packet,
		ap_uint<3> position, ap_uint<64> item) {
#pragma HLS INLINE
	switch ((unsigned)position) {
	case 0: packet.range(63, 0) = item; break;
	case 1: packet.range(127, 64) = item; break;
	case 2: packet.range(191, 128) = item; break;
	case 3: packet.range(255, 192) = item; break;
	case 4: packet.range(319, 256) = item; break;
	case 5: packet.range(383, 320) = item; break;
	case 6: packet.range(447, 384) = item; break;
	default: packet.range(511, 448) = item; break;
	}
	return packet;
}

void tapa_merge_ordered_select(
		tapa::istream<TapaBankedMergeItem>& input0,
		tapa::istream<TapaBankedMergeItem>& input1,
		tapa::istream<TapaBankedMergeItem>& input2,
		tapa::istream<TapaBankedMergeItem>& input3,
		tapa::istream<TapaBankedMergeItem>& input4,
		tapa::istream<TapaBankedMergeItem>& input5,
		tapa::istream<TapaBankedMergeItem>& input6,
		tapa::istream<TapaBankedMergeItem>& input7,
		tapa::ostream<TapaMergeSparsePacket>& output) {
#pragma HLS INLINE off
	TapaBankedMergeItem head[8];
	bool valid[8];
	bool ended[8];
#pragma HLS ARRAY_PARTITION variable=head complete
#pragma HLS ARRAY_PARTITION variable=valid complete
#pragma HLS ARRAY_PARTITION variable=ended complete
	for (int bank = 0; bank < 8; ++bank) {
#pragma HLS UNROLL
		valid[bank] = false;
		ended[bank] = false;
	}
	bool have_active_row = false;
	bool have_active_bank = false;
	id_t active_row = 0;
	bool active_heavy = false;
	ap_uint<3> active_bank = 0;
	ap_uint<8> active_mask = 0;
	while (!(ended[0] && ended[1] && ended[2] && ended[3]
			&& ended[4] && ended[5] && ended[6] && ended[7])) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=1048576 avg=4096
#define TAPA_MERGE_REFILL(B, STREAM) \
		if (!valid[B] && !ended[B]) { \
			const TapaBankedMergeItem item = STREAM.read(); \
			if (item.last) ended[B] = true; \
			else { head[B] = item; valid[B] = true; } \
		}
		TAPA_MERGE_REFILL(0, input0)
		TAPA_MERGE_REFILL(1, input1)
		TAPA_MERGE_REFILL(2, input2)
		TAPA_MERGE_REFILL(3, input3)
		TAPA_MERGE_REFILL(4, input4)
		TAPA_MERGE_REFILL(5, input5)
		TAPA_MERGE_REFILL(6, input6)
		TAPA_MERGE_REFILL(7, input7)
#undef TAPA_MERGE_REFILL
		const bool any_valid = valid[0] || valid[1] || valid[2] || valid[3]
			|| valid[4] || valid[5] || valid[6] || valid[7];
		if (!any_valid) continue;
		if (!have_active_row) {
			if (have_active_bank) {
				active_heavy = head[(unsigned)active_bank].heavy;
				active_mask = active_heavy ? ap_uint<8>(0xff)
					: (ap_uint<8>)(1 << active_bank);
				have_active_row = true;
				have_active_bank = false;
				continue;
			}
			ap_uint<35> row_key[8];
#pragma HLS ARRAY_PARTITION variable=row_key complete
			for (int bank = 0; bank < 8; ++bank) {
#pragma HLS UNROLL
				ap_uint<35> key = 0;
				key.range(34, 3) = head[bank].row;
				key.range(2, 0) = bank;
				row_key[bank] = valid[bank] ? key : ~ap_uint<35>(0);
			}
			const ap_uint<35> min01 = row_key[0] < row_key[1]
				? row_key[0] : row_key[1];
			const ap_uint<35> min23 = row_key[2] < row_key[3]
				? row_key[2] : row_key[3];
			const ap_uint<35> min45 = row_key[4] < row_key[5]
				? row_key[4] : row_key[5];
			const ap_uint<35> min67 = row_key[6] < row_key[7]
				? row_key[6] : row_key[7];
			const ap_uint<35> min03 = min01 < min23 ? min01 : min23;
			const ap_uint<35> min47 = min45 < min67 ? min45 : min67;
			const ap_uint<35> minimum = min03 < min47 ? min03 : min47;
			active_row = minimum.range(34, 3);
			active_bank = minimum.range(2, 0);
			have_active_bank = true;
			continue;
		}

		ap_uint<14> key[8];
#pragma HLS ARRAY_PARTITION variable=key complete
		for (int bank = 0; bank < 8; ++bank) {
#pragma HLS UNROLL
			// A ROW_CONTEXT row exists only in its selected physical root.  A
			// HEAVY row inserts EOR into all roots, and the per-root reducers keep
			// row order.  Therefore active_mask alone proves row membership; a
			// redundant 32-bit equality here was the 7.953-ns selector path.
			const bool belongs = valid[bank] && active_mask[bank];
			key[bank] = belongs
				? (head[bank].eor ? (ap_uint<14>)(1 << 13)
					: (ap_uint<14>)(head[bank].col >> 3))
				: ~ap_uint<14>(0);
		}
		const ap_uint<14> min01 = key[0] < key[1] ? key[0] : key[1];
		const ap_uint<14> min23 = key[2] < key[3] ? key[2] : key[3];
		const ap_uint<14> min45 = key[4] < key[5] ? key[4] : key[5];
		const ap_uint<14> min67 = key[6] < key[7] ? key[6] : key[7];
		const ap_uint<14> min03 = min01 < min23 ? min01 : min23;
		const ap_uint<14> min47 = min45 < min67 ? min45 : min67;
		const ap_uint<14> minimum_key = min03 < min47 ? min03 : min47;
		TapaMergeSparsePacket packet;
		packet.row = active_row;
		packet.items = 0;
		packet.mask = 0;
		packet.products = 0;
		packet.errors = 0;
		packet.eor = minimum_key[13];
		packet.heavy = active_heavy;
		packet.last = false;
		if (packet.eor) {
			ap_uint<8> consume_mask = 0;
			for (int bank = 0; bank < 8; ++bank) {
#pragma HLS UNROLL
				consume_mask[bank] = valid[bank] && active_mask[bank]
					&& head[bank].eor;
				if (consume_mask[bank]) {
					packet.errors |= head[bank].errors;
					if (packet.products == 0)
						packet.products = head[bank].products;
					else if (packet.products != head[bank].products)
						packet.errors |= ap_uint<2>(2);
					valid[bank] = false;
				}
			}
			const ap_uint<4> eor_count = tapa_popcount8(consume_mask);
			if (eor_count != (active_heavy ? ap_uint<4>(8) : ap_uint<4>(1)))
				packet.errors |= ap_uint<2>(2);
			output.write(packet);
			have_active_row = false;
			continue;
		}
		const ap_uint<13> quotient = minimum_key.range(12, 0);
		bool select[8];
#pragma HLS ARRAY_PARTITION variable=select complete
		for (int bank = 0; bank < 8; ++bank) {
#pragma HLS UNROLL
			select[bank] = valid[bank] && active_mask[bank]
				&& !head[bank].eor
				&& (ap_uint<13>)(head[bank].col >> 3) == quotient;
		}
		const ap_uint<3> salt = (quotient ^ (quotient >> 3)) & 7;
		for (int low = 0; low < 8; ++low) {
#pragma HLS UNROLL
			const ap_uint<3> bank = active_heavy
				? (ap_uint<3>)low ^ salt : active_bank;
			const bool found = select[(unsigned)bank]
				&& (active_heavy
					? (ap_uint<3>)(head[(unsigned)bank].col & 7)
						== (ap_uint<3>)low
					: low == 0);
			if (found) {
				ap_uint<64> item = 0;
				item.range(31, 0) = head[(unsigned)bank].col;
				item.range(63, 32) = head[(unsigned)bank].value;
				packet.items.range(low * 64 + 63, low * 64) = item;
				packet.mask[low] = 1;
			}
		}
		for (int bank = 0; bank < 8; ++bank) {
#pragma HLS UNROLL
			if (select[bank]) valid[bank] = false;
		}
		output.write(packet);
	}
	TapaMergeSparsePacket terminal;
	terminal.row = 0;
	terminal.items = 0;
	terminal.mask = 0;
	terminal.products = 0;
	terminal.errors = 0;
	terminal.eor = false;
	terminal.heavy = false;
	terminal.last = true;
	output.write(terminal);
}

void tapa_merge_ordered_compact(tapa::istream<TapaMergeSparsePacket>& input,
		tapa::ostream<TapaMergeCompactPacket>& output) {
#pragma HLS INLINE off
	bool done = false;
	while (!done) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=1048576 avg=4096
		const TapaMergeSparsePacket source = input.read();
		TapaMergeCompactPacket result;
		result.row = source.row;
		result.items = 0;
		result.count = 0;
		result.products = source.products;
		result.errors = source.errors;
		result.eor = source.eor;
		result.heavy = source.heavy;
		result.last = source.last;
		for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
			if (source.mask[lane]) {
				const ap_uint<64> item
					= source.items.range(lane * 64 + 63, lane * 64);
				result.items = tapa_merge_insert64(
					result.items, result.count, item);
				result.count++;
			}
		}
		output.write(result);
		done = source.last;
	}
}

void tapa_merge_ordered_emit(tapa::istream<TapaMergeCompactPacket>& input,
		tapa::ostream<TapaMergeResultPacket>& output) {
#pragma HLS INLINE off
	bool done = false;
	while (!done) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=1048576 avg=4096
		const TapaMergeCompactPacket source = input.read();
		TapaMergeResultPacket packet = 0;
		packet.range(511, 0) = source.items;
		packet.range(515, 512) = source.count;
		packet.range(547, 516) = source.row;
		packet[548] = source.eor;
		packet[549] = source.last;
		packet.range(551, 550) = source.errors;
		packet[552] = source.heavy;
		packet.range(584, 553) = source.products;
		output.write(packet);
		done = source.last;
	}
}
