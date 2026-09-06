# Balance the eight BRAM-heavy local-MERGE tasks across the three U280 SLRs.
#
# The unconstrained routed checkpoint placed local-MERGE BRAM tiles as
#   SLR0 / SLR1 / SLR2 = 366 / 396 / 214.
# Shards 1 and 7 were split across SLR1/SLR2, while shard 4 (the source of six
# of the ten worst setup paths) was wholly in the congested SLR1.  Collecting
# shards 1, 4, and 7 in SLR2 yields the intended 366 / 244 / 366 balance and
# removes the internal SLR crossings from shards 1 and 7.

set shard_targets {
  SLR0 {2 3 6}
  SLR1 {0 5}
  SLR2 {1 4 7}
}
set shard_distribution "3/2/3"
if {[info exists ::env(TAPA_SHARD_LAYOUT)] && $::env(TAPA_SHARD_LAYOUT) eq "3_1_4"} {
  # Post-place diagnostics from the quiet-drain run showed the central DENSE
  # reducer and shard 5 sharing a level-6 congestion window in SLR1, while
  # SLR2 remained at only 41.25% CLB and 59.38% BRAM-tile utilization.  Keep
  # shard 0 beside the reducer and move shard 5 to SLR2 for reducer-heavy
  # successors such as bank-wide extraction.
  set shard_targets {
    SLR0 {2 3 6}
    SLR1 {0}
    SLR2 {1 4 5 7}
  }
  set shard_distribution "3/1/4"
}
if {[info exists ::env(TAPA_SHARD_LAYOUT)] && $::env(TAPA_SHARD_LAYOUT) eq "2_1_5"} {
  # The 3/1/4 placement still put 93.23% of the BRAM tiles in SLR0 and
  # consumed 63.48% of the SLR0<->SLR1 super-long lines.  Move only shard 2
  # from SLR0 to the under-used side of the device.  SLR1 keeps shard 0, so
  # the central allocator/reducer is not given another BRAM-heavy neighbor.
  set shard_targets {
    SLR0 {3 6}
    SLR1 {0}
    SLR2 {1 2 4 5 7}
  }
  set shard_distribution "2/1/5"
}

set assigned 0
foreach {slr shard_indices} $shard_targets {
  set pblock_name "pblock_dynamic_${slr}"
  set pblock [get_pblocks -quiet $pblock_name]
  if {[llength $pblock] != 1} {
    error "expected exactly one $pblock_name, found [llength $pblock]"
  }
  if {[get_property IS_SOFT $pblock] ne "0"} {
    error "$pblock_name must be a hard placement pblock"
  }

  foreach shard_index $shard_indices {
    set pattern [format {^.*/tapa_shard_local_merge_%d$} $shard_index]
    set task [get_cells -quiet -hier -regexp $pattern]
    if {[llength $task] != 1} {
      error "expected exactly one local-MERGE shard $shard_index, found [llength $task]"
    }
    add_cells_to_pblock $pblock $task
    set memberships [get_pblocks -quiet -of_objects $task]
    set membership_names [get_property NAME $memberships]
    if {[lsearch -exact $membership_names $pblock_name] < 0} {
      error "local-MERGE shard $shard_index was not assigned to $pblock_name"
    }
    puts "SLR_BALANCE_ASSIGN shard=$shard_index slr=$slr cell=[get_property NAME $task] pblock=$pblock_name"
    incr assigned
  }
}

if {$assigned != 8} {
  error "expected eight local-MERGE assignments, emitted $assigned"
}
puts "SLR_BALANCE_COMPLETE assigned=$assigned distribution=$shard_distribution"
