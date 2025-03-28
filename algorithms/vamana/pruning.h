#pragma once

#include "parlay/primitives.h"
#include <utility>

namespace parlayANN {

enum class PruningHeuristic {
  ARYA_MOUNT,                   // Arya and Mount's heuristic
  ROBUST_PRUNE,                 // Robust Prune
  ARYA_MOUNT_SANITY_CHECK,      // Arya and Mount's heuristic with sanity check
  NEAREST_M,                    // Regular KNN graph
  FURTHEST_M,                   // Weird KNN graph
  MEDIAN_ADAPTIVE,              // Median Adaptive
  TOP_MEDIAN_ADAPTIVE,          // Top Median Adaptive
  MEAN_SORTED_BASELINE,         // Mean Sorted Baseline
  QUANTILE_NOT_MIN,             // Quantile Not Min
  ARYA_MOUNT_REVERSED,          // Arya and Mount's heuristic reversed
  PROBABILISTIC_RANK,           // Probabilistic Rank
  NEIGHBORHOOD_OVERLAP,         // Neighborhood Overlap
  CHEAP_OUTDEGREE_CONDITIONAL,  // Cheap Outdegree Conditional
  LARGE_OUTDEGREE_CONDITIONAL,  // Large Outdegree Conditional
  GEOMETRIC_MEAN,               // Geometric Mean
  SIGMOID_RATIO,                // Sigmoid Ratio
  SIGMOID_RATIO_STEEPNESS_5,    // Sigmoid Ratio Steepness 5
  SIGMOID_RATIO_STEEPNESS_10,   // Sigmoid Ratio Steepness 10
  ARYA_MOUNT_SHUFFLED,          // Arya and Mount's heuristic shuffled
  ARYA_MOUNT_RANDOM_ON_REJECTS, // Arya and Mount's heuristic with random on
                                // rejects (1% chance of including rejects)
  ARYA_MOUNT_RANDOM_ON_REJECTS_5,  // Arya and Mount's heuristic with random on
                                   // rejects (5% chance of including rejects)
  ARYA_MOUNT_RANDOM_ON_REJECTS_10, // Arya and Mount's heuristic with random on
                                   // rejects (10% chance of including rejects)
  ARYA_MOUNT_SIGMOID_ON_REJECTS,   // Arya and Mount's heuristic with sigmoid on
                                   // rejects
  ARYA_MOUNT_SIGMOID_ON_REJECTS_STEEPNESS_5, // Arya and Mount's heuristic with
                                             // sigmoid on rejects (steepness 5)
  CHEAP_OUTDEGREE_CONDITIONAL_2,
  CHEAP_OUTDEGREE_CONDITIONAL_4,
  CHEAP_OUTDEGREE_CONDITIONAL_6,
  CHEAP_OUTDEGREE_CONDITIONAL_8,
  CHEAP_OUTDEGREE_CONDITIONAL_10,
  CHEAP_OUTDEGREE_CONDITIONAL_12,
  CHEAP_OUTDEGREE_CONDITIONAL_14,
  CHEAP_OUTDEGREE_CONDITIONAL_M,
  ONE_SPANNER,
  ARYA_MOUNT_PLUS_SPANNER,
};

template <typename PointRange, typename QPointRange, typename indexType>
struct PruningHeuristicSelector {
  using Point = typename PointRange::Point;
  using QPoint = typename QPointRange::Point;
  using GraphI = Graph<indexType>;
  using PR = PointRange;
  PruningHeuristic _heuristic;

  PruningHeuristicSelector(PruningHeuristic heuristic)
      : _heuristic(heuristic) {}

  std::pair<parlay::sequence<indexType>, long>
  pruneCandidateSet(indexType p, parlay::sequence<pid> &cand, GraphI &G,
                    PR &Points, double alpha, bool add = true) {
    switch (_heuristic) {
    case PruningHeuristic::ARYA_MOUNT:
      return aryaMount(p, cand, G, Points, alpha, add);
    case PruningHeuristic::ROBUST_PRUNE:
      return robustPrune(p, cand, G, Points, alpha, add);
    case PruningHeuristic::ARYA_MOUNT_SANITY_CHECK:
      return aryaMount(p, cand, G, Points, alpha, add);
    case PruningHeuristic::NEAREST_M:
    default:
      return std::make_pair(parlay::sequence<indexType>(), 0);
    }
  }

  std::pair<parlay::sequence<indexType>, long>
  robustPrune(indexType p, parlay::sequence<pid> &cand, GraphI &G, PR &Points,
              double alpha, bool add = true) {}

  std::pair<parlay::sequence<indexType>, long>
  aryaMount(indexType p, parlay::sequence<pid> &cand, GraphI &G, PR &Point,
            double alpha, bool add = true) {}
}

} // namespace parlayANN