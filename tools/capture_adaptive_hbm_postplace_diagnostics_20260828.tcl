# Lightweight diagnostics captured after place_design and before phys_opt_design.
# Keep the implementation alive even if an optional report is unavailable in
# this Vivado release; the marker records every command's return code.
set diag_dir [get_property DIRECTORY [current_run]]
# Vivado 2022.2 can return a project-relative value such as
# "./.runs/impl_1" here even though PRE hooks execute with the implementation
# run directory as their working directory.  Reusing that relative value would
# incorrectly create paths below impl_1/.runs/impl_1.  The hook is always
# sourced by an implementation step, so its working directory is the reliable
# fallback for a non-absolute run-directory property.
if {$diag_dir eq "" || [file pathtype $diag_dir] ne "absolute"} {
  set diag_dir [file normalize [pwd]]
} else {
  set diag_dir [file normalize $diag_dir]
}
if {![file isdirectory $diag_dir]} {
  error "post-place diagnostic directory does not exist: $diag_dir"
}

set timing_summary [file join $diag_dir postplace_timing_summary_for_reject.rpt]
set timing_paths [file join $diag_dir postplace_worst100_for_reject.rpt]
set slr_util [file join $diag_dir postplace_slr_util_for_reject.rpt]
set congestion [file join $diag_dir postplace_congestion_for_reject.rpt]
set high_fanout [file join $diag_dir postplace_high_fanout_for_reject.rpt]
set marker [file join $diag_dir .postplace_diagnostics_complete]

set rc_timing_summary [catch {
  report_timing_summary -delay_type min_max -max_paths 20 -file $timing_summary
} msg_timing_summary]
set rc_timing_paths [catch {
  report_timing -delay_type max -max_paths 100 -sort_by group \
    -path_type full_clock_expanded -file $timing_paths
} msg_timing_paths]
set rc_slr_util [catch {
  report_utilization -slr -file $slr_util
} msg_slr_util]
set rc_congestion [catch {
  report_design_analysis -congestion -file $congestion
} msg_congestion]
set rc_high_fanout [catch {
  report_high_fanout_nets -fanout_greater_than 500 -max_nets 200 \
    -file $high_fanout
} msg_high_fanout]

set fp [open $marker w]
puts $fp "timing_summary_rc=$rc_timing_summary message=$msg_timing_summary"
puts $fp "timing_paths_rc=$rc_timing_paths message=$msg_timing_paths"
puts $fp "slr_util_rc=$rc_slr_util message=$msg_slr_util"
puts $fp "congestion_rc=$rc_congestion message=$msg_congestion"
puts $fp "high_fanout_rc=$rc_high_fanout message=$msg_high_fanout"
close $fp
puts "POSTPLACE_DIAGNOSTICS_COMPLETE marker=$marker"
