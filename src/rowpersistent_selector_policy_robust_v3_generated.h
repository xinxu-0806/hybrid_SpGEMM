#ifndef ROWPERSISTENT_SELECTOR_POLICY_ROBUST_V3_H
#define ROWPERSISTENT_SELECTOR_POLICY_ROBUST_V3_H

// Generated from all 38 representative validation matrices after three
// explicitly recorded exclusions; held-out tests were not accessed.
// Decisive mixed winners (>= 2.0%): Muite/Chebyshev4
// Maximum admitted-path validation regret: 1.0%.
#include <cstdint>

namespace rowpersistent_selector_policy {

enum Decision : uint8_t { kMergeOnly = 0, kRowAdaptive = 1, kDenseOnly = 2 };

struct AdaptiveRegion {
  double min_work;
  double max_work;
  uint32_t min_dense_rows;
  double min_merge_dense_ratio;
};

constexpr char kPolicySha256[] = "1737f10aa25bd51d61a19494eb3a6a701241f1fc6a8f82624791c08eca2e9210";
constexpr bool kDenseEnabled = false;
constexpr double kDenseMinWork = 0;
constexpr double kDenseMinRowFraction = 0;
constexpr uint32_t kAdaptiveRegionCount = 1;
constexpr AdaptiveRegion kAdaptiveRegions[2] = {
  {0.10000000000000001, 0.59999999999999998, 8U, 16},
  {0, 0, 1U, 0},
};

inline Decision decide(double dense_work_fraction,
                       double dense_row_fraction,
                       uint64_t merge_rows, uint64_t dense_rows) {
  if (kDenseEnabled && dense_work_fraction >= kDenseMinWork
      && dense_row_fraction >= kDenseMinRowFraction)
    return kDenseOnly;
  for (uint32_t index = 0; index < kAdaptiveRegionCount; ++index) {
    const AdaptiveRegion& region = kAdaptiveRegions[index];
    if (dense_rows >= region.min_dense_rows
        && dense_work_fraction >= region.min_work
        && dense_work_fraction <= region.max_work
        && static_cast<double>(merge_rows)
            >= region.min_merge_dense_ratio * static_cast<double>(dense_rows))
      return kRowAdaptive;
  }
  return kMergeOnly;
}

}  // namespace rowpersistent_selector_policy

#endif  // ROWPERSISTENT_SELECTOR_POLICY_ROBUST_V3_H
