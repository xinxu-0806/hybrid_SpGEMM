#include "adaptive_hbm_merge_forward_tree8_tapa.h"

namespace {

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

static ap_uint<3> popcount4(ap_uint<4> value) {
#pragma HLS INLINE
	return (value[0] + value[1]) + (value[2] + value[3]);
}

static ap_uint<35> item_key(ap_uint<128> item) {
#pragma HLS INLINE
	ap_uint<35> key = 0;
	key.range(34, 19) = item.range(109, 94);
	key[18] = item[93];
	key.range(17, 2) = item.range(15, 0);
	key.range(1, 0) = 0;
	return key;
}

static bool same_item(ap_uint<128> lhs, ap_uint<128> rhs) {
#pragma HLS INLINE
	return lhs.range(109, 94) == rhs.range(109, 94)
		&& lhs[93] == rhs[93]
		&& lhs.range(15, 0) == rhs.range(15, 0);
}

static MergeForwardTree8Token terminal_token() {
#pragma HLS INLINE
	MergeForwardTree8Token token = 0;
	token[128] = 1;
	return token;
}

static void write_by_level(ap_uint<128> item, ap_uint<2> node_level,
		tapa::ostream<MergeForwardTree8Token>& exit,
		tapa::ostream<MergeForwardTree8Token>& upward) {
#pragma HLS INLINE
	MergeForwardTree8Token token = 0;
	token.range(127, 0) = item;
	if (item.range(92, 91) == node_level) exit.write(token);
	else upward.write(token);
}

}  // namespace

void merge_forward_tree8_read(tapa::mmap<const ap_uint<128> > leaf_in,
		id_t count, tapa::ostream<MergeForwardTree8Token>& output) {
	for (id_t cursor = 0; cursor < count; ++cursor) {
#pragma HLS PIPELINE II=1
		MergeForwardTree8Token token = 0;
		token.range(127, 0) = leaf_in[cursor];
		output.write(token);
	}
	output.write(terminal_token());
}

void merge_forward_tree8_leaf(
		tapa::istream<MergeForwardTree8Token>& input,
		tapa::ostream<MergeForwardTree8Token>& direct,
		tapa::ostream<MergeForwardTree8Token>& upward) {
	bool done = false;
	while (!done) {
#pragma HLS PIPELINE II=1
		MergeForwardTree8Token token;
		if (!input.try_read(token)) continue;
		if (token[128]) {
			direct.write(token);
			upward.write(token);
			done = true;
		} else if (token.range(92, 91) == 0) {
			direct.write(token);
		} else {
			upward.write(token);
		}
	}
}

// Merge equal keys and forward the completed row from the node named by its
// allocation.  Rows allocated above this node continue upward.  The child
// streams contain the same allocation sequence whenever this node is owned;
// explicit empty-branch EORs close rounded-up buddy allocations.
void merge_forward_tree8_node(
		tapa::istream<MergeForwardTree8Token>& input0,
		tapa::istream<MergeForwardTree8Token>& input1,
		ap_uint<2> node_level,
		tapa::ostream<MergeForwardTree8Token>& exit,
		tapa::ostream<MergeForwardTree8Token>& upward) {
	ap_uint<128> head0 = 0;
	ap_uint<128> head1 = 0;
	bool valid0 = false;
	bool valid1 = false;
	bool ended0 = false;
	bool ended1 = false;
	while (!(ended0 && ended1)) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=1048576 avg=4096
		if (!valid0 && !ended0) {
			MergeForwardTree8Token token;
			if (input0.try_read(token)) {
				if (token[128]) ended0 = true;
				else { head0 = token.range(127, 0); valid0 = true; }
			}
		}
		if (!valid1 && !ended1) {
			MergeForwardTree8Token token;
			if (input1.try_read(token)) {
				if (token[128]) ended1 = true;
				else { head1 = token.range(127, 0); valid1 = true; }
			}
		}
		if ((!valid0 && !ended0) || (!valid1 && !ended1)
				|| (!valid0 && !valid1)) continue;
		const ap_uint<35> key0 = valid0 ? item_key(head0)
			: ~ap_uint<35>(0);
		const ap_uint<35> key1 = valid1 ? item_key(head1)
			: ~ap_uint<35>(0);
		const bool choose0 = key0 <= key1;
		const ap_uint<128> selected = choose0 ? head0 : head1;
		const bool consume0 = valid0 && same_item(head0, selected);
		const bool consume1 = valid1 && same_item(head1, selected);
		const float value0 = consume0 && !selected[93]
			? from_raw(head0.range(47, 16)) : 0.0f;
		const float value1 = consume1 && !selected[93]
			? from_raw(head1.range(47, 16)) : 0.0f;
		const float sum = value0 + value1;
#pragma HLS BIND_OP variable=sum op=fadd impl=fulldsp latency=5
		if (consume0) valid0 = false;
		if (consume1) valid1 = false;
		ap_uint<128> result = selected;
		if (selected[93]) {
			result.range(47, 16) = 0;
			write_by_level(result, node_level, exit, upward);
		} else if (nonzero(sum)) {
			result.range(47, 16) = raw_float(sum);
			write_by_level(result, node_level, exit, upward);
		}
	}
	exit.write(terminal_token());
	upward.write(terminal_token());
}

void merge_forward_tree8_drain(
		tapa::istream<MergeForwardTree8Token>& input) {
	bool done = false;
	while (!done) {
#pragma HLS PIPELINE II=1
		done = input.read()[128];
	}
}

// Four independent collectors model the four numerical HBM output paths.
// Each cycle can preserve one token from every attached exit; no global
// minimum/ordered selector is reintroduced after the flexible tree.
void merge_forward_tree8_collect4(
		tapa::istream<MergeForwardTree8Token>& input0,
		tapa::istream<MergeForwardTree8Token>& input1,
		tapa::istream<MergeForwardTree8Token>& input2,
		tapa::istream<MergeForwardTree8Token>& input3,
		tapa::mmap<ap_uint<512> > data_out,
		tapa::mmap<ap_uint<8> > mask_out,
		tapa::mmap<id_t> stats_out) {
	bool ended[4] = {false, false, false, false};
#pragma HLS ARRAY_PARTITION variable=ended complete
	id_t words = 0;
	id_t items = 0;
	id_t eors = 0;
	while (!(ended[0] && ended[1] && ended[2] && ended[3])) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=1048576 avg=4096
		ap_uint<512> word = 0;
		ap_uint<8> mask = 0;
		ap_uint<4> eor_mask = 0;
#define MF_COLLECT(S, STREAM) \
		if (!ended[S]) { \
			MergeForwardTree8Token token; \
			if (STREAM.try_read(token)) { \
				if (token[128]) ended[S] = true; \
				else { \
					word.range(S * 128 + 127, S * 128) = token.range(127, 0); \
					mask[S] = 1; \
					eor_mask[S] = token[93]; \
				} \
			} \
		}
		MF_COLLECT(0, input0)
		MF_COLLECT(1, input1)
		MF_COLLECT(2, input2)
		MF_COLLECT(3, input3)
#undef MF_COLLECT
		items += popcount4(mask.range(3, 0));
		eors += popcount4(eor_mask);
		if (mask != 0) {
			data_out[words] = word;
			mask_out[words] = mask;
			++words;
		}
	}
	stats_out[0] = words;
	stats_out[1] = items;
	stats_out[2] = eors;
}

void merge_forward_tree8_terminal(
		tapa::ostream<MergeForwardTree8Token>& output) {
	output.write(terminal_token());
}

void adaptive_hbm_merge_forward_tree8_tapa(
		tapa::mmap<const ap_uint<128> > leaf0, id_t count0,
		tapa::mmap<const ap_uint<128> > leaf1, id_t count1,
		tapa::mmap<const ap_uint<128> > leaf2, id_t count2,
		tapa::mmap<const ap_uint<128> > leaf3, id_t count3,
		tapa::mmap<const ap_uint<128> > leaf4, id_t count4,
		tapa::mmap<const ap_uint<128> > leaf5, id_t count5,
		tapa::mmap<const ap_uint<128> > leaf6, id_t count6,
		tapa::mmap<const ap_uint<128> > leaf7, id_t count7,
		tapa::mmap<ap_uint<512> > output0,
		tapa::mmap<ap_uint<512> > output1,
		tapa::mmap<ap_uint<512> > output2,
		tapa::mmap<ap_uint<512> > output3,
		tapa::mmap<ap_uint<8> > masks0,
		tapa::mmap<ap_uint<8> > masks1,
		tapa::mmap<ap_uint<8> > masks2,
		tapa::mmap<ap_uint<8> > masks3,
		tapa::mmap<id_t> statistics0,
		tapa::mmap<id_t> statistics1,
		tapa::mmap<id_t> statistics2,
		tapa::mmap<id_t> statistics3) {
	tapa::streams<MergeForwardTree8Token, 8, 8> memory_to_leaf("memory_to_leaf");
	tapa::streams<MergeForwardTree8Token, 8, 8> leaf_up("leaf_up");
	tapa::streams<MergeForwardTree8Token, 8, 8> leaf_exit("leaf_exit");
	tapa::streams<MergeForwardTree8Token, 4, 8> level1_up("level1_up");
	tapa::streams<MergeForwardTree8Token, 4, 8> level1_exit("level1_exit");
	tapa::streams<MergeForwardTree8Token, 2, 8> level2_up("level2_up");
	tapa::streams<MergeForwardTree8Token, 2, 8> level2_exit("level2_exit");
	tapa::stream<MergeForwardTree8Token, 8> root_exit("root_exit");
	tapa::stream<MergeForwardTree8Token, 8> root_up("root_up");
	tapa::stream<MergeForwardTree8Token, 2> empty_exit("empty_exit");

	tapa::task()
		.invoke(merge_forward_tree8_read, leaf0, count0, memory_to_leaf[0])
		.invoke(merge_forward_tree8_read, leaf1, count1, memory_to_leaf[1])
		.invoke(merge_forward_tree8_read, leaf2, count2, memory_to_leaf[2])
		.invoke(merge_forward_tree8_read, leaf3, count3, memory_to_leaf[3])
		.invoke(merge_forward_tree8_read, leaf4, count4, memory_to_leaf[4])
		.invoke(merge_forward_tree8_read, leaf5, count5, memory_to_leaf[5])
		.invoke(merge_forward_tree8_read, leaf6, count6, memory_to_leaf[6])
		.invoke(merge_forward_tree8_read, leaf7, count7, memory_to_leaf[7])
		.invoke<tapa::join, 8>(merge_forward_tree8_leaf,
			memory_to_leaf, leaf_exit, leaf_up)
		.invoke(merge_forward_tree8_node, leaf_up[0], leaf_up[1], 1,
			level1_exit[0], level1_up[0])
		.invoke(merge_forward_tree8_node, leaf_up[2], leaf_up[3], 1,
			level1_exit[1], level1_up[1])
		.invoke(merge_forward_tree8_node, leaf_up[4], leaf_up[5], 1,
			level1_exit[2], level1_up[2])
		.invoke(merge_forward_tree8_node, leaf_up[6], leaf_up[7], 1,
			level1_exit[3], level1_up[3])
		.invoke(merge_forward_tree8_node, level1_up[0], level1_up[1], 2,
			level2_exit[0], level2_up[0])
		.invoke(merge_forward_tree8_node, level1_up[2], level1_up[3], 2,
			level2_exit[1], level2_up[1])
		.invoke(merge_forward_tree8_node, level2_up[0], level2_up[1], 3,
			root_exit, root_up)
		.invoke(merge_forward_tree8_drain, root_up)
		.invoke(merge_forward_tree8_terminal, empty_exit)
		.invoke(merge_forward_tree8_collect4, leaf_exit[0], leaf_exit[4],
			level1_exit[0], level2_exit[0], output0, masks0, statistics0)
		.invoke(merge_forward_tree8_collect4, leaf_exit[1], leaf_exit[5],
			level1_exit[1], level2_exit[1], output1, masks1, statistics1)
		.invoke(merge_forward_tree8_collect4, leaf_exit[2], leaf_exit[6],
			level1_exit[2], root_exit, output2, masks2, statistics2)
		.invoke(merge_forward_tree8_collect4, leaf_exit[3], leaf_exit[7],
			level1_exit[3], empty_exit, output3, masks3, statistics3);
}
