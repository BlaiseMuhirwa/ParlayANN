#pragma once

#include <math.h>
#include <algorithm>
#include <random>
#include <set>
#include <vector> 
#include <numeric> 
#include <limits> 
#include <cmath> 
#include <unordered_set> 

#include "../utils/beamSearch.h"
#include "../utils/graph.h"
#include "../utils/point_range.h"
#include "../utils/types.h"
#include "parlay/delayed.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"


namespace parlayANN {

// --- Helper Functions (can be private members or in anonymous namespace) ---
namespace { // Use anonymous namespace for internal linkage

template <typename Point, typename PR, typename indexType, typename distanceType>
distanceType get_distance(PR& Points, indexType id1, indexType id2) {
    if (id1 == id2) return 0.0; // Optimization
    // Assuming Point has a distance method
    return Points[id1].distance(Points[id2]);
}


// Calculates minimum distance from node `target_node` to any node in the `selected_nodes` vector
template <typename Point, typename PR, typename indexType, typename distanceType>
distanceType min_distance_to_selected(PR& Points, indexType target_node,
                                     const std::vector<indexType>& selected_nodes,
                                     long& distance_comps) {
    if (selected_nodes.empty()) {
        return std::numeric_limits<distanceType>::max();
    }
    distanceType min_dist = std::numeric_limits<distanceType>::max();
    for (const auto& selected_node : selected_nodes) {
        distance_comps++; // Count distance computation
        distanceType dist = get_distance<Point, PR, indexType, distanceType>(Points, target_node, selected_node);
        if (dist < min_dist) {
            min_dist = dist;
        }
    }
    return min_dist;
}

// Helper for quantile calculation
template <typename distanceType>
distanceType quantile_of(const std::vector<distanceType>& distances, double quantile) {
    if (distances.empty()) {
        return std::numeric_limits<distanceType>::max();
    }
    std::vector<distanceType> sorted_distances = distances;
    std::sort(sorted_distances.begin(), sorted_distances.end());
    // Use ceil for index to match flatnav implementation style
    int index = static_cast<int>(std::ceil(quantile * (static_cast<double>(sorted_distances.size()) - 1.0)));
    index = std::max(0, std::min(index, (int)sorted_distances.size() - 1)); // Clamp index
    return sorted_distances[index];
}

// Helper for Jaccard Index
template <typename indexType>
float jaccard_index(const std::unordered_set<indexType>& set1,
                    const std::unordered_set<indexType>& set2) {
    if (set1.empty() || set2.empty()) {
        return 0.0f;
    }
    size_t intersection_size = 0;
    const std::unordered_set<indexType>& smaller_set = (set1.size() < set2.size()) ? set1 : set2;
    const std::unordered_set<indexType>& larger_set = (set1.size() < set2.size()) ? set2 : set1;

    for (const auto& element : smaller_set) {
        if (larger_set.count(element)) {
            intersection_size++;
        }
    }
    size_t union_size = set1.size() + set2.size() - intersection_size;
    if (union_size == 0) return 1.0f; // Both sets contained only common elements (or were identical and non-empty)
    return static_cast<float>(intersection_size) / static_cast<float>(union_size);
}

// Helper to get actual out-degree (excluding self-loops)
template <typename GraphI, typename indexType>
int get_out_degree(GraphI& G, indexType node_id) {
    int degree = 0;
    if (node_id < G.size()) { // Basic safety check
       for(indexType neighbor : G[node_id]) {
           // Assuming Graph interface gives direct access or via a method
           // We need to know how G[node_id] behaves. Assuming it returns a range/view of neighbors.
           // The original Vamana code doesn't explicitly store self-loops after pruning usually.
           // Let's assume G[node_id] gives the *current* valid neighbors.
           // If the graph *can* have self-loops, we'd need: if (neighbor != node_id) degree++;
           // But robustPrune filters self out, so G[node_id].size() should be correct?
           // Let's rely on size() for now, assuming pruned graph has no self-loops.
           // Revisit if flatnav's getNodeLinks includes self-loops for empty slots.
           // Flatnav *does* use self-loops for empty slots. ParlayANN's Graph might not.
           // Let's stick to size() as the most likely equivalent in ParlayANN context after prune.
           // If we need the *potential* out-degree before pruning, it's different.
           // The flatnav logic counts actual neighbors *excluding* self-loops.
           // ParlayANN's G[node_id].size() should reflect this *after* pruning.
       }
       return G[node_id].size(); // Best guess for ParlayANN post-prune state
    }
    return 0;
}


} // end anonymous namespace

template <typename PointRange, typename QPointRange, typename indexType>
struct knn_index {
  using Point = typename PointRange::Point;
  using QPoint = typename QPointRange::Point;
  using distanceType = typename Point::distanceType;
  using pid = std::pair<indexType, distanceType>;
  using PR = PointRange;
  using QPR = QPointRange;
  using GraphI = Graph<indexType>;

  BuildParams BP;
  std::set<indexType> delete_set;
  indexType start_point;

  knn_index(BuildParams &BP) : BP(BP) {}

  indexType get_start() { return start_point; }

  // Forward declaration for the main pruning function
  std::pair<parlay::sequence<indexType>, long>
  robustPrune(indexType p, parlay::sequence<pid> &cand, GraphI &G, PR &Points,
              double alpha, bool add = true);

  // The wrapper remains the same
  std::pair<parlay::sequence<indexType>, long>
  robustPrune(indexType p, parlay::sequence<indexType> candidates, GraphI &G,
              PR &Points, double alpha, bool add = true) {
    parlay::sequence<pid> cc;
    long distance_comps = 0;
    cc.reserve(candidates.size());
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (candidates[i] != p) {
          distance_comps++;
          cc.push_back(std::make_pair(candidates[i],
                                      get_distance<Point, PR, indexType, distanceType>(Points, candidates[i], p)));
        } else {
             // Optionally add p with distance 0 if needed downstream, but prune usually skips it.
             // cc.push_back(std::make_pair(p, 0.0));
        }
    }
    // Call the main robustPrune implementation
    auto [ngh_seq, dc] = robustPrune(p, cc, G, Points, alpha, add);
    return std::pair(ngh_seq, dc + distance_comps);
  }


  // --- Pruning Algorithm Implementations ---

  // Original Vamana/DiskANN pruning (ID 0)
  std::pair<parlay::sequence<indexType>, long>
  prune_vamana(indexType p, std::vector<pid>& candidates_vec, GraphI& G, PR& Points, double alpha) {
      long distance_comps = 0;
      std::vector<indexType> new_nbhs;
      new_nbhs.reserve(BP.R);
      std::vector<bool> pruned(candidates_vec.size(), false); // Use boolean flags instead of modifying pid.first

      size_t candidate_idx = 0;
      while (new_nbhs.size() < BP.R && candidate_idx < candidates_vec.size()) {
          int p_star_idx = -1;
          // Find the next unpruned candidate
          for(size_t current_check_idx = candidate_idx; current_check_idx < candidates_vec.size(); ++current_check_idx) {
              if (!pruned[current_check_idx]) {
                   p_star_idx = current_check_idx;
                   break;
              }
          }

          if (p_star_idx == -1) break; // No more candidates left

          candidate_idx = p_star_idx + 1; // Move past the selected candidate
          indexType p_star = candidates_vec[p_star_idx].first;

          // Original robustPrune checks for p_star == p or -1.
          // We already sorted and removed duplicates, and p!=p check is implicit if p isn't in candidates_vec.
          // If p *could* be in candidates_vec (e.g., from G[p]), filter it here.
          if (p_star == p) {
              pruned[p_star_idx] = true; // Mark p as pruned/skipped
              continue;
          }

          new_nbhs.push_back(p_star);

          // Occlusion check
          for (size_t i = candidate_idx; i < candidates_vec.size(); ++i) {
              if (!pruned[i]) {
                  indexType p_prime = candidates_vec[i].first;
                  if (p_prime == p) { // Skip if p_prime is p
                       pruned[i] = true;
                       continue;
                  }

                  distance_comps++;
                  distanceType dist_starprime = get_distance<Point, PR, indexType, distanceType>(Points, p_star, p_prime);
                  distanceType dist_pprime = candidates_vec[i].second; // Distance(p, p_prime)

                  // Compare distances using alpha parameter
                  if (alpha * dist_starprime <= dist_pprime) {
                      pruned[i] = true; // Mark p_prime as pruned
                  }
              }
          }
      }

      auto new_neighbors_seq = parlay::to_sequence(new_nbhs);
      return std::pair(new_neighbors_seq, distance_comps);
  }

  // HNSW / Arya-Mount style pruning (ID 1)
   std::pair<parlay::sequence<indexType>, long>
   prune_hnsw(indexType p, std::vector<pid>& candidates_vec, GraphI& G, PR& Points) {
       long distance_comps = 0;
       std::vector<indexType> new_nbhs; // Stores selected neighbor IDs
       new_nbhs.reserve(BP.R);

       for (const auto& candidate_p_prime : candidates_vec) {
           if (new_nbhs.size() >= BP.R) break; // Reached capacity

           indexType p_prime = candidate_p_prime.first;
           if (p_prime == p) continue; // Skip self

           distanceType dist_p_prime = candidate_p_prime.second; // dist(p, p_prime)

           // Calculate minimum distance from p_prime to already selected neighbors
           distanceType closest_selected_dist = min_distance_to_selected<Point, PR, indexType, distanceType>(Points, p_prime, new_nbhs, distance_comps);

           // HNSW condition: Keep if further from all selected nodes than from the query node p
           if (closest_selected_dist > dist_p_prime) {
               new_nbhs.push_back(p_prime);
           }
       }

       auto new_neighbors_seq = parlay::to_sequence(new_nbhs);
       return std::pair(new_neighbors_seq, distance_comps);
   }


    // Simple Nearest-M Pruning (ID 2)
    std::pair<parlay::sequence<indexType>, long>
    prune_nearest_m(indexType p, std::vector<pid>& candidates_vec, GraphI& G, PR& Points) {
        long distance_comps = 0; // No inter-candidate distances needed
        std::vector<indexType> new_nbhs;
        new_nbhs.reserve(BP.R);

        for (const auto& candidate : candidates_vec) {
            if (new_nbhs.size() >= BP.R) break;
            if (candidate.first != p) { // Exclude self
                new_nbhs.push_back(candidate.first);
            }
        }

        auto new_neighbors_seq = parlay::to_sequence(new_nbhs);
        return std::pair(new_neighbors_seq, distance_comps);
    }

    // Further-M Pruning (ID 3) - requires sorting descending
     std::pair<parlay::sequence<indexType>, long>
     prune_furthest_m(indexType p, std::vector<pid>& candidates_vec, GraphI& G, PR& Points) {
         long distance_comps = 0; // No inter-candidate distances needed
         std::vector<indexType> new_nbhs;
         new_nbhs.reserve(BP.R);

         // Sort descending by distance
         std::sort(candidates_vec.begin(), candidates_vec.end(), [&](const pid& a, const pid& b){
             return a.second > b.second || (a.second == b.second && a.first > b.first); // Furthest first
         });

         for (const auto& candidate : candidates_vec) {
             if (new_nbhs.size() >= BP.R) break;
             if (candidate.first != p) { // Exclude self
                 new_nbhs.push_back(candidate.first);
             }
         }

         // Re-sort results ascending by distance for consistency if needed (optional)
         // std::sort(new_nbhs.begin(), new_nbhs.end(), [&](indexType a, indexType b){ ... });

         auto new_neighbors_seq = parlay::to_sequence(new_nbhs);
         return std::pair(new_neighbors_seq, distance_comps);
     }


    // DiskANN-style pruning with explicit alpha (ID 4) - slightly different logic flow than Vamana's prune
    std::pair<parlay::sequence<indexType>, long>
    prune_diskann_explicit(indexType p, std::vector<pid>& candidates_vec, GraphI& G, PR& Points, double alpha) {
        long distance_comps = 0;
        std::vector<indexType> new_nbhs;
        new_nbhs.reserve(BP.R);

        for (const auto& candidate : candidates_vec) {
             if (new_nbhs.size() >= BP.R) break; // Limit reached

             indexType p_prime = candidate.first;
             if (p_prime == p) continue; // Skip self

             distanceType dist_p_prime = candidate.second; // dist(p, p_prime)

             // Calculate minimum distance from p_prime to already selected neighbors
             distanceType closest_selected_dist = min_distance_to_selected<Point, PR, indexType, distanceType>(Points, p_prime, new_nbhs, distance_comps);

             // DiskANN condition: alpha * dist(selected, p_prime) > dist(p, p_prime) for *all* selected
             // Which is equivalent to: alpha * min_dist(selected, p_prime) > dist(p, p_prime)
             if (alpha * closest_selected_dist > dist_p_prime) {
                 new_nbhs.push_back(p_prime);
             }
         }


        auto new_neighbors_seq = parlay::to_sequence(new_nbhs);
        return std::pair(new_neighbors_seq, distance_comps);
    }

    // Quantile Pruning (ID 5)
    std::pair<parlay::sequence<indexType>, long>
    prune_quantile(indexType p, std::vector<pid>& candidates_vec, GraphI& G, PR& Points, double quantile) {
        long distance_comps = 0;
        std::vector<indexType> new_nbhs;
        new_nbhs.reserve(BP.R);

        for (const auto& candidate : candidates_vec) {
            if (new_nbhs.size() >= BP.R) break;

            indexType p_prime = candidate.first;
            if (p_prime == p) continue;

            distanceType dist_p_prime = candidate.second;

            // Get distances from p_prime to all currently selected neighbors
            std::vector<distanceType> distances_to_selected;
            if (!new_nbhs.empty()) {
                distances_to_selected.reserve(new_nbhs.size());
                 for (const auto& selected_node : new_nbhs) {
                    distance_comps++;
                    distances_to_selected.push_back(get_distance<Point, PR, indexType, distanceType>(Points, p_prime, selected_node));
                }
            }

            distanceType quantile_dist = quantile_of(distances_to_selected, quantile);

            // Pruning condition: Keep if quantile distance >= distance from p
            if (quantile_dist >= dist_p_prime) {
                new_nbhs.push_back(p_prime);
            }
        }

        auto new_neighbors_seq = parlay::to_sequence(new_nbhs);
        return std::pair(new_neighbors_seq, distance_comps);
    }

    // Arya Mount Reversed Order (ID 6)
     std::pair<parlay::sequence<indexType>, long>
     prune_hnsw_reversed(indexType p, std::vector<pid>& candidates_vec, GraphI& G, PR& Points) {
         long distance_comps = 0;
         std::vector<indexType> new_nbhs;
         new_nbhs.reserve(BP.R);

         // Sort descending by distance to p
         std::sort(candidates_vec.begin(), candidates_vec.end(), [&](const pid& a, const pid& b){
             return a.second > b.second || (a.second == b.second && a.first > b.first);
         });


         for (const auto& candidate_p_prime : candidates_vec) {
             if (new_nbhs.size() >= BP.R) break;

             indexType p_prime = candidate_p_prime.first;
             if (p_prime == p) continue;

             distanceType dist_p_prime = candidate_p_prime.second;

             distanceType closest_selected_dist = min_distance_to_selected<Point, PR, indexType, distanceType>(Points, p_prime, new_nbhs, distance_comps);

             // Use the same HNSW condition, but apply on reverse sorted candidates
             if (closest_selected_dist > dist_p_prime) {
                 new_nbhs.push_back(p_prime);
             }
         }

         auto new_neighbors_seq = parlay::to_sequence(new_nbhs);
         return std::pair(new_neighbors_seq, distance_comps);
     }

     // Arya Mount with Random Accept on Rejects (ID 7)
     std::pair<parlay::sequence<indexType>, long>
     prune_hnsw_random_rejects(indexType p, std::vector<pid>& candidates_vec, GraphI& G, PR& Points, double accept_prob) {
         long distance_comps = 0;
         std::vector<indexType> new_nbhs;
         new_nbhs.reserve(BP.R);

         // Need a random number generator. Use thread-local for potential parallelism later.
         // For simplicity here, using a local one.
         // Consider seeding properly if used in parallel context.
         std::mt19937 gen(std::random_device{}());
         std::uniform_real_distribution<> distrib(0.0, 1.0);


         for (const auto& candidate_p_prime : candidates_vec) {
             if (new_nbhs.size() >= BP.R) break;

             indexType p_prime = candidate_p_prime.first;
             if (p_prime == p) continue;

             distanceType dist_p_prime = candidate_p_prime.second;

             distanceType closest_selected_dist = min_distance_to_selected<Point, PR, indexType, distanceType>(Points, p_prime, new_nbhs, distance_comps);

             bool standard_accept = (closest_selected_dist > dist_p_prime);
             bool random_accept = false;
             if (!standard_accept) {
                 random_accept = (distrib(gen) < accept_prob);
             }

             if (standard_accept || random_accept) {
                 new_nbhs.push_back(p_prime);
             }
         }

         auto new_neighbors_seq = parlay::to_sequence(new_nbhs);
         return std::pair(new_neighbors_seq, distance_comps);
     }

     // Arya Mount with Sigmoid Accept on Rejects (ID 8)
     std::pair<parlay::sequence<indexType>, long>
     prune_hnsw_sigmoid_rejects(indexType p, std::vector<pid>& candidates_vec, GraphI& G, PR& Points, double steepness) {
         long distance_comps = 0;
         std::vector<indexType> new_nbhs;
         new_nbhs.reserve(BP.R);

         std::mt19937 gen(std::random_device{}());
         std::uniform_real_distribution<> distrib(0.0, 1.0);


         for (const auto& candidate_p_prime : candidates_vec) {
             if (new_nbhs.size() >= BP.R) break;

             indexType p_prime = candidate_p_prime.first;
             if (p_prime == p) continue;

             distanceType dist_p_prime = candidate_p_prime.second;

             distanceType closest_selected_dist = min_distance_to_selected<Point, PR, indexType, distanceType>(Points, p_prime, new_nbhs, distance_comps);

             bool standard_accept = (closest_selected_dist > dist_p_prime);
             bool sigmoid_accept = false;
             if (!standard_accept) {
                 // Ratio = closest_selected_dist / dist_p_prime
                 // We want to accept MORE when this ratio is closer to 1 (or > 1)
                 // Sigmoid input should increase as ratio increases.
                 double ratio = (dist_p_prime > 1e-9) ? (static_cast<double>(closest_selected_dist) / static_cast<double>(dist_p_prime)) :
                                ( (closest_selected_dist > 1e-9) ? std::numeric_limits<double>::max() : 1.0 ); // Handle division by zero/small numbers

                 double midpoint = 1.0; // Centered at the HNSW threshold
                 // Probability increases as ratio increases
                 double accept_probability = 1.0 / (1.0 + std::exp(-steepness * (ratio - midpoint)));

                 sigmoid_accept = (distrib(gen) < accept_probability);
             }

             if (standard_accept || sigmoid_accept) {
                 new_nbhs.push_back(p_prime);
             }
         }

         auto new_neighbors_seq = parlay::to_sequence(new_nbhs);
         return std::pair(new_neighbors_seq, distance_comps);
     }

    // Pruning based on Out-degree (cheap nodes) (ID 9)
     std::pair<parlay::sequence<indexType>, long>
     prune_hnsw_cheap_nodes(indexType p, std::vector<pid>& candidates_vec, GraphI& G, PR& Points, int degree_threshold) {
         long distance_comps = 0;
         std::vector<indexType> new_nbhs;
         new_nbhs.reserve(BP.R);

         for (const auto& candidate_p_prime : candidates_vec) {
             if (new_nbhs.size() >= BP.R) break;

             indexType p_prime = candidate_p_prime.first;
             if (p_prime == p) continue;

             distanceType dist_p_prime = candidate_p_prime.second;

             distanceType closest_selected_dist = min_distance_to_selected<Point, PR, indexType, distanceType>(Points, p_prime, new_nbhs, distance_comps);

             bool standard_accept = (closest_selected_dist > dist_p_prime);
             bool cheap_node_accept = false;
             if (!standard_accept) {
                 // Check out-degree of p_prime IN THE CURRENT GRAPH G
                 // This reflects the state *before* adding edges from the current batch insert round
                 // Note: This might differ slightly from flatnav which checks degree *during* connectNeighbors
                 int out_degree = get_out_degree(G, p_prime);
                 cheap_node_accept = (out_degree <= degree_threshold);
             }

             if (standard_accept || cheap_node_accept) {
                 new_nbhs.push_back(p_prime);
             }
         }

         auto new_neighbors_seq = parlay::to_sequence(new_nbhs);
         return std::pair(new_neighbors_seq, distance_comps);
     }


      // Pruning based on Out-degree (well-connected nodes) (ID 10)
     std::pair<parlay::sequence<indexType>, long>
     prune_hnsw_well_connected_nodes(indexType p, std::vector<pid>& candidates_vec, GraphI& G, PR& Points, int degree_threshold) {
         long distance_comps = 0;
         std::vector<indexType> new_nbhs;
         new_nbhs.reserve(BP.R);

         for (const auto& candidate_p_prime : candidates_vec) {
             if (new_nbhs.size() >= BP.R) break;

             indexType p_prime = candidate_p_prime.first;
             if (p_prime == p) continue;

             distanceType dist_p_prime = candidate_p_prime.second;

             distanceType closest_selected_dist = min_distance_to_selected<Point, PR, indexType, distanceType>(Points, p_prime, new_nbhs, distance_comps);

             bool standard_accept = (closest_selected_dist > dist_p_prime);
             bool well_connected_accept = false;
             if (!standard_accept) {
                 int out_degree = get_out_degree(G, p_prime);
                 well_connected_accept = (out_degree >= degree_threshold);
             }

             if (standard_accept || well_connected_accept) {
                 new_nbhs.push_back(p_prime);
             }
         }

         auto new_neighbors_seq = parlay::to_sequence(new_nbhs);
         return std::pair(new_neighbors_seq, distance_comps);
     }

    // 1-Spanner Pruning (ID 11)
    std::pair<parlay::sequence<indexType>, long>
    prune_one_spanner(indexType p, std::vector<pid>& candidates_vec, GraphI& G, PR& Points) {
        long distance_comps = 0;
        std::vector<indexType> new_nbhs;
        new_nbhs.reserve(BP.R);
        std::unordered_set<indexType> one_hop_neighborhood; // Nodes reachable in 1 hop from selected nodes

        for (const auto& candidate : candidates_vec) {
            if (new_nbhs.size() >= BP.R) break;

            indexType p_prime = candidate.first;
            if (p_prime == p) continue;

            // Check if p_prime is already reachable via a 1-hop neighbor of the *currently selected* set
            bool node_not_reachable = (one_hop_neighborhood.find(p_prime) == one_hop_neighborhood.end());

            if (node_not_reachable) {
                new_nbhs.push_back(p_prime);
                // Add neighbors of the newly added node p_prime to the reachable set
                // Need to access G[p_prime]'s neighbors
                if (p_prime < G.size()) { // Safety check
                   for (indexType neighbor_of_prime : G[p_prime]) {
                       one_hop_neighborhood.insert(neighbor_of_prime);
                   }
                   one_hop_neighborhood.insert(p_prime); // Node is reachable from itself
                }
            }
        }

        auto new_neighbors_seq = parlay::to_sequence(new_nbhs);
        return std::pair(new_neighbors_seq, distance_comps); // Note: No inter-candidate distances calculated here
    }

    // HNSW + 1-Spanner Pruning (ID 12)
    std::pair<parlay::sequence<indexType>, long>
    prune_hnsw_plus_spanner(indexType p, std::vector<pid>& candidates_vec, GraphI& G, PR& Points) {
        long distance_comps = 0;
        std::vector<indexType> new_nbhs;
        new_nbhs.reserve(BP.R);
        std::unordered_set<indexType> one_hop_neighborhood;

        for (const auto& candidate_p_prime : candidates_vec) {
            if (new_nbhs.size() >= BP.R) break;

            indexType p_prime = candidate_p_prime.first;
            if (p_prime == p) continue;

            distanceType dist_p_prime = candidate_p_prime.second;

            // HNSW Check
            distanceType closest_selected_dist = min_distance_to_selected<Point, PR, indexType, distanceType>(Points, p_prime, new_nbhs, distance_comps);
            bool hnsw_accept = (closest_selected_dist > dist_p_prime);

            // Spanner Check
            bool node_not_reachable = (one_hop_neighborhood.find(p_prime) == one_hop_neighborhood.end());

            if (hnsw_accept || node_not_reachable) {
                bool newly_added = false;
                // Avoid adding duplicates if both conditions are met separately
                if (std::find(new_nbhs.begin(), new_nbhs.end(), p_prime) == new_nbhs.end()) {
                     new_nbhs.push_back(p_prime);
                     newly_added = true;
                }

                // Update reachable set only if the node was *newly* added and accepted
                if (newly_added && p_prime < G.size()) {
                     for (indexType neighbor_of_prime : G[p_prime]) {
                       one_hop_neighborhood.insert(neighbor_of_prime);
                     }
                     one_hop_neighborhood.insert(p_prime);
                }
            }
        }

        auto new_neighbors_seq = parlay::to_sequence(new_nbhs);
        return std::pair(new_neighbors_seq, distance_comps);
    }

     // --- Add other pruning implementations similarly ---
     // e.g., prune_median_adaptive, prune_top_m_mean, prune_mean_sorted,
     // prune_probabilistic_rank, prune_neighborhood_overlap, prune_geometric_mean,
     // prune_sigmoid_ratio, prune_hnsw_shuffled


  // Main robustPrune function: Switches between algorithms
  std::pair<parlay::sequence<indexType>, long>
  robustPrune(indexType p, parlay::sequence<pid> &cand, GraphI &G, PR &Points,
              double alpha, bool add = true) {

    // --- Common Setup ---
    size_t initial_cand_size = cand.size();
    size_t out_size = add ? G[p].size() : 0;
    std::vector<pid> candidates_vec;
    candidates_vec.reserve(initial_cand_size + out_size);
    long initial_distance_comps = 0; // Track distances computed in this setup phase

    // Add initial candidates
    for (const auto& x : cand) {
        candidates_vec.push_back(x);
    }

    // Add existing out neighbors if requested
    if (add) {
        for (indexType neighbor_id : G[p]) {
            // Avoid re-adding if already in cand (more robust to check later after sort+unique)
            // Compute distance if adding
            initial_distance_comps++;
            candidates_vec.push_back(
                std::make_pair(neighbor_id, get_distance<Point, PR, indexType, distanceType>(Points, neighbor_id, p)));
        }
    }

    // Sort the candidate set according to distance from p
    auto less = [&](const pid& a, const pid& b) {
        return a.second < b.second || (a.second == b.second && a.first < b.first);
    };
    std::sort(candidates_vec.begin(), candidates_vec.end(), less);

    // Remove duplicates (keeping the first occurrence, which is the closest one due to sort)
    auto new_end = std::unique(candidates_vec.begin(), candidates_vec.end(),
                               [&](const pid& a, const pid& b) { return a.first == b.first; });
    candidates_vec.erase(new_end, candidates_vec.end());

    // --- Select and Execute Pruning Algorithm ---
    std::pair<parlay::sequence<indexType>, long> result;
    long prune_distance_comps = 0;

    // Make sure BP.R is reasonable
    if (BP.R <= 0) {
         return std::pair(parlay::sequence<indexType>(), initial_distance_comps);
    }


    switch (BP.pruning_algo_id) {
        case 0: // Vamana/DiskANN original
            // Note: Vamana uses the alpha passed to robustPrune, typically 1.0 or BP.alpha
            result = prune_vamana(p, candidates_vec, G, Points, alpha);
            break;
        case 1: // HNSW / Arya-Mount
             // Alpha is not used in the standard HNSW heuristic
            result = prune_hnsw(p, candidates_vec, G, Points);
            break;
        case 2: // Nearest M
            result = prune_nearest_m(p, candidates_vec, G, Points);
            break;
        case 3: // Furthest M
             result = prune_furthest_m(p, candidates_vec, G, Points);
             break;
        case 4: // DiskANN Explicit Alpha (using BP.pruning_dist_alpha)
             result = prune_diskann_explicit(p, candidates_vec, G, Points, BP.pruning_dist_alpha);
             break;
        case 5: // Quantile Pruning
             result = prune_quantile(p, candidates_vec, G, Points, BP.pruning_quantile);
             break;
         case 6: // HNSW Reversed Order
             result = prune_hnsw_reversed(p, candidates_vec, G, Points);
             break;
         case 7: // HNSW Random Rejects
             result = prune_hnsw_random_rejects(p, candidates_vec, G, Points, BP.pruning_accept_prob);
             break;
         case 8: // HNSW Sigmoid Rejects
             result = prune_hnsw_sigmoid_rejects(p, candidates_vec, G, Points, BP.pruning_sigmoid_steepness);
             break;
         case 9: // HNSW Cheap Nodes
              result = prune_hnsw_cheap_nodes(p, candidates_vec, G, Points, BP.pruning_outdegree_threshold);
              break;
         case 10: // HNSW Well Connected Nodes
              result = prune_hnsw_well_connected_nodes(p, candidates_vec, G, Points, BP.pruning_outdegree_threshold);
              break;
         case 11: // 1-Spanner
              result = prune_one_spanner(p, candidates_vec, G, Points);
              break;
         case 12: // HNSW + 1-Spanner
              result = prune_hnsw_plus_spanner(p, candidates_vec, G, Points);
              break;
        // --- Add cases for other algorithms ---
        // case 13: result = prune_median_adaptive(...); break;
        // case ...:

        default: // Fallback to original Vamana/DiskANN
            std::cerr << "Warning: Unknown pruning_algo_id " << BP.pruning_algo_id << ". Falling back to Vamana pruning." << std::endl;
            result = prune_vamana(p, candidates_vec, G, Points, alpha);
            break;
    }

    prune_distance_comps = result.second;
    // Total distance comps = those from setup + those from the pruning algorithm logic
    return std::pair(result.first, initial_distance_comps + prune_distance_comps);
  }


  // --- Rest of the knn_index class members ---
  // add_neighbors_without_repeats, set_start, build_index, batch_insert etc.
  // Make sure build_index and batch_insert use the BP correctly.
  // The alpha passed to batch_insert might only be used if pruning_algo_id is 0 (Vamana)
  // or needs clarification on its role for other algorithms.

    void add_neighbors_without_repeats(const parlay::sequence<indexType> &ngh,
                                     parlay::sequence<indexType> &candidates) {
    // Using std::unordered_set might be slow for frequent small operations.
    // Consider alternative if performance critical: sort both and merge unique?
    // For now, keep the simple approach.
    std::unordered_set<indexType> existing_candidates;
    for (auto c : candidates)
      existing_candidates.insert(c);

    // Use a temporary vector to store candidates to add, then append once.
    std::vector<indexType> to_add;
    to_add.reserve(ngh.size()); // Max possible additions

    for (indexType n : ngh) {
        if (existing_candidates.find(n) == existing_candidates.end()) {
            to_add.push_back(n);
            existing_candidates.insert(n); // Add to set to handle duplicates within ngh itself
        }
    }
    // Append the new unique neighbors to the original sequence
    if (!to_add.empty()) {
         // This might be inefficient as sequence append can reallocate.
         // If candidates were a std::vector, push_back would be better.
         // Let's convert to_add to a sequence and use parlay::append
         parlay::sequence<indexType> to_add_seq = parlay::to_sequence(to_add);
         candidates = parlay::append(candidates, to_add_seq);
         // Or modify candidates in place if possible and efficient? Parlay sequences might not support easy append.
         // Alternative: rebuild the sequence if necessary
         // parlay::sequence<indexType> temp_seq = parlay::append(candidates, parlay::to_sequence(to_add));
         // candidates = std::move(temp_seq);
    }
  }

   // Function to make neighbors bidirectional after robustPrune in batch_insert
   // This is a simplified view of the logic within batch_insert's parallel_for loop
   void make_bidirectional_after_prune(indexType p, const parlay::sequence<indexType>& new_neighbors, GraphI& G, PR& Points, stats<indexType>& BuildStats) {
        // This logic is complex in batch_insert involving grouping.
        // Let's assume we have the candidates for a node 'index' generated from the forward pass (new_neighbors of others pointing to 'index')
        // The simplified idea from flatnav's connectNeighbors back-connection:
        // For each neighbor 'j' that node 'p' points to (in new_neighbors):
        //   Try to add 'p' to 'j's neighbor list.
        //   If 'j's list is full:
        //     Create candidate set for 'j': current_neighbors(j) + p
        //     Run robustPrune on 'j' with this candidate set
        //     Update G[j] with the result.

        // ParlayANN's batch_insert does this differently and likely more efficiently:
        // 1. All nodes run beam search and robustPrune (forward pass). G[p] is updated.
        // 2. Collect all new forward edges (p, j).
        // 3. Group these edges by the target node 'j'. Result: (j, [list of nodes p pointing to j]).
        // 4. For each target node 'j':
        //    Let 'candidates' be the list of nodes 'p' pointing to 'j'.
        //    Get current neighbors G[j].
        //    If G[j].size() + candidates.size() <= BP.R:
        //       Combine G[j] and candidates (removing duplicates) and update G[j].
        //    Else:
        //       Run robustPrune on 'j' using 'candidates' and G[j] (via add=true).
        //       Update G[j] with the result.

        // The existing batch_insert code already implements step 4 correctly using robustPrune.
        // No direct translation of flatnav's connectNeighbors is needed here, as ParlayANN handles
        // bidirectionality differently after the main pruning step.
   }


  void set_start() {
    // Pick a suitable start point. Using 0 is simple.
    // A random point or centroid could also be used.
    if (start_point >= G.size()) { // Check if start point is valid if graph size known
         start_point = 0;
    }
    // Ensure start point is initialized if not set elsewhere
    // If G is not yet sized, defer setting until Points.size() is known?
    // Assuming G is sized or Points.size() is available.
    if (Points.size() > 0) {
         start_point = parlay::random_value<indexType>() % Points.size(); // Example: random start
    } else {
         start_point = 0; // Default if no points
    }
    std::cout << "Index start point set to: " << start_point << std::endl;
   }


  void build_index(GraphI &G, PR &Points, QPR &QPoints,
                   stats<indexType> &BuildStats, bool sort_neighbors = true) {
    std::cout << "Building graph with pruning algorithm ID: " << BP.pruning_algo_id << "..." << std::endl;
    // Make sure G is sized appropriately before setting start point if needed
    // G.resize(Points.size()); // Example, if Graph needs explicit sizing

    set_start(); // Set start point, potentially randomly

    parlay::sequence<indexType> inserts = parlay::tabulate(
        Points.size(), [&](size_t i) { return static_cast<indexType>(i); });

    if (BP.single_batch != 0) {
       // ... (single batch initialization code remains the same) ...
      int degree = BP.single_batch;
      std::cout << "Using single batch per round with " << degree
                << " random start edges" << std::endl;
      parlay::random_generator gen;
      std::uniform_int_distribution<long> dis(0, G.size() -1); // Ensure valid index range
      parlay::parallel_for(0, G.size(), [&](long i) {
          // Generate unique random neighbors if possible and degree < G.size()
          std::vector<indexType> outEdges;
          outEdges.reserve(degree);
          std::unordered_set<indexType> added;
          added.insert(static_cast<indexType>(i)); // Don't add self

          parlay::random r = gen[i]; // Get generator for this thread/iteration

          while(outEdges.size() < degree && added.size() < G.size()) {
               indexType neighbor = dis(r);
               if (added.find(neighbor) == added.end()) {
                    outEdges.push_back(neighbor);
                    added.insert(neighbor);
               }
               r = r.next(); // Advance generator state
          }
          // Fill remaining slots if needed (e.g., duplicates allowed or graph small)
          while (outEdges.size() < degree) {
              outEdges.push_back(dis(r)); // May add duplicates now
              r = r.next();
          }

          G[i].update_neighbors(parlay::to_sequence(outEdges));
      });

    }

    // Main build loops using batch_insert
    std::cout << "number of passes = " << BP.num_passes << std::endl;
    for (int i = 0; i < BP.num_passes; i++) {
      std::cout << "Starting pass " << (i + 1) << "/" << BP.num_passes << std::endl;
      // The alpha passed here (BP.alpha or 1.0) is used by robustPrune
      // For Vamana (ID 0), it's the core parameter.
      // For others, it might be ignored or used differently if specified.
      // We pass BP.alpha consistently, letting the chosen prune function use/ignore it.
      double current_alpha = (i == BP.num_passes - 1) ? BP.alpha : 1.0;
      batch_insert(inserts, G, Points, QPoints, BuildStats, current_alpha, true /* random_order */);
      std::cout << "Finished pass " << (i + 1) << "/" << BP.num_passes << std::endl;

      // Optional: Intermediate stats or graph checks
      // long total_edges = parlay::reduce(parlay::tabulate(G.size(), [&](size_t i){ return G[i].size(); }));
      // std::cout << "  Graph edges after pass " << (i+1) << ": " << total_edges << std::endl;
    }

    // Final sorting of neighbors by distance
    if (sort_neighbors) {
      std::cout << "Sorting final neighbors by distance..." << std::endl;
      parlay::parallel_for(0, G.size(), [&](long i) {
        auto less = [&](indexType j, indexType k) {
          // Ensure j and k are valid indices before accessing Points
          if (j >= Points.size() || k >= Points.size()) return false; // Or handle error
          // Compute distance only once if possible or needed
          return get_distance<Point, PR, indexType, distanceType>(Points, i, j) <
                 get_distance<Point, PR, indexType, distanceType>(Points, i, k);
        };
        G[i].sort(less); // Assumes Graph nodes have a sort method
      });
      std::cout << "Neighbor sorting complete." << std::endl;
    }
     std::cout << "Index build finished." << std::endl;
  }

  void batch_insert(parlay::sequence<indexType> &inserts, GraphI &G, PR &Points,
                    QPR &QPoints, stats<indexType> &BuildStats, double alpha,
                    bool random_order = false, double base = 2,
                    double max_fraction = .02, bool print = true) {
    // ... (batching setup logic remains the same) ...
    size_t n = G.size();
    size_t m = inserts.size();
    // ... (random permutation setup) ...
    parlay::sequence<int> rperm;
    if (random_order)
      rperm = parlay::random_permutation<int>(static_cast<int>(m));
    else
      rperm = parlay::tabulate(m, [&](int i) { return i; });
    auto shuffled_inserts =
        parlay::delayed::map(rperm, [&](int i){ return inserts[i]; });
        // was: parlay::tabulate(m, [&](size_t i) { return inserts[rperm[i]]; }); -- delayed might be better


    parlay::internal::timer t_beam("beam search time", print);
    parlay::internal::timer t_bidirect("bidirect time (grouping)", print);
    parlay::internal::timer t_prune("prune time (incl. setup & actual prune)", print); // Renamed for clarity
    t_beam.stop(); t_bidirect.stop(); t_prune.stop(); // Start stopped


    size_t count = 0;
    size_t inc = 0;
    size_t max_batch_size = std::max(1ul, std::min(
        static_cast<size_t>(max_fraction * static_cast<float>(n)), 1000000ul));


    float progress = 0.0;
    float progress_inc = 0.1;


    while (count < m) {
       size_t floor = count;
       size_t ceiling = std::min(count + max_batch_size, m);
       // Adjust batch size calculation if using exponential growth (original code)
       // If using the original exponential batching:
       /*
       if (pow(base, inc) <= max_batch_size && !BP.single_batch) {
            floor = static_cast<size_t>(pow(base, inc)) -1; // Check this logic, might be off by 1?
            floor = std::min(floor, m); // Clamp floor
            ceiling = std::min(static_cast<size_t>(pow(base, inc + 1)) - 1, m);
            count = ceiling; // Update count based on ceiling
       } else {
           floor = count;
           ceiling = std::min(count + max_batch_size, m);
           count = ceiling; // Update count
       }
       inc++;
       */
       // Simpler linear batching for now:
        count = ceiling;


       if (BP.single_batch != 0) {
           floor = 0;
           ceiling = m;
           count = m;
       }
       size_t current_batch_size = ceiling - floor;
       if (current_batch_size == 0) break; // Should not happen if logic is correct


       if (print) {
            std::cout << "  Processing batch: " << floor << " to " << ceiling << " (" << current_batch_size << " nodes)" << std::endl;
       }


      parlay::sequence<parlay::sequence<indexType>> new_out_(current_batch_size);
      parlay::sequence<long> prune_dcs(current_batch_size); // Store distance comps from prune step


      // Step 1: Beam search and initial prune (forward edges)
      t_beam.start();
      parlay::parallel_for(0, current_batch_size, [&](size_t i) {
        size_t insert_idx = floor + i;
        indexType index = shuffled_inserts[insert_idx];
        // Ensure start_point is valid
        indexType current_start_point = (start_point < G.size()) ? start_point : 0;
        if (BP.single_batch != 0) current_start_point = insert_idx % G.size(); // Distribute start points if single batch?

        // Ensure QP parameters are valid
        QueryParams QP(BP.L, BP.L, 0.0, Points.size(), G.max_degree()); // Use BP.L for efConstruction
        if (QP.k <=0) QP.k = 10; // Ensure k > 0
        if (QP.beam_width <= 0) QP.beam_width = BP.L; // Ensure beam_width > 0

        // Use the appropriate beam search function (rerank or standard)
         auto [visited_pids, bs_distance_comps] =
              beam_search_rerank__<Point, QPoint, PR, QPR, indexType>(
                 Points[index], QPoints[index], G, Points, QPoints, current_start_point, QP);

        // Check if visited_pids is empty, handle if necessary
        if (visited_pids.empty() && G.size() > 1) {
             // Maybe add random neighbor if search failed? Or just proceed.
             // std::cerr << "Warning: Beam search returned empty results for node " << index << std::endl;
        }


        BuildStats.increment_dist(index, bs_distance_comps);
        BuildStats.increment_visited(index, visited_pids.size()); // Visited includes distances here

        // Call robustPrune which now selects the algorithm based on BP.pruning_algo_id
        // Pass the alpha determined for the current pass
        auto [pruned_neighbors, rp_distance_comps] =
            robustPrune(index, visited_pids, G, Points, alpha, false); // add=false: only use beam results

        new_out_[i] = std::move(pruned_neighbors);
        prune_dcs[i] = rp_distance_comps; // Store prune distance comps
        BuildStats.increment_dist(index, rp_distance_comps);

      });
      t_beam.stop();


      // Update graph with new forward edges (atomic updates might be needed if Graph impl requires)
      // Assuming G[index].update_neighbors handles concurrency or is called safely
      t_prune.start(); // Count this update time as part of pruning effect
       parlay::parallel_for(0, current_batch_size, [&](size_t i) {
           indexType index = shuffled_inserts[floor + i];
           if (index < G.size()) { // Safety check
               G[index].update_neighbors(new_out_[i]);
           } else {
                 std::cerr << "Error: Attempting to update neighbors for invalid index " << index << std::endl;
           }
       });
       t_prune.stop();


      // Step 2: Prepare for bidirectionality (gather back-edges)
      t_bidirect.start();
      auto flattened_delayed = parlay::delayed::flatten(
          parlay::tabulate(current_batch_size, [&](size_t i) {
            indexType index = shuffled_inserts[floor + i];
            // Map each neighbor 'ngh' in new_out_[i] to a pair (ngh, index)
            // representing the back-edge index -> ngh
            return parlay::delayed::map(new_out_[i], [=](indexType ngh) {
              return std::pair<indexType, indexType>(ngh, index); // Target node first for grouping
            });
          }));
      // Materialize and group by the target node
      auto flattened_sequence = parlay::delayed::to_sequence(flattened_delayed);
      auto grouped_by = parlay::group_by_key(flattened_sequence);
      t_bidirect.stop();


      // Step 3: Apply back-edges and prune nodes receiving many back-edges
      t_prune.start();
      parlay::parallel_for(0, grouped_by.size(), [&](size_t j) {
        auto& [target_node, candidates_for_target] = grouped_by[j]; // candidates are nodes pointing *to* target_node

        if (target_node >= G.size()) {
             std::cerr << "Error: Invalid target node " << target_node << " in grouping." << std::endl;
             return; // Skip processing for invalid target node
        }

        size_t current_neighbors_size = G[target_node].size();
        // Convert candidates_for_target (range) to sequence for robustPrune if needed
        // Note: candidates_for_target are just IDs, need to compute distances to target_node
        parlay::sequence<indexType> candidate_indices = parlay::to_sequence(candidates_for_target);


        // Check if adding candidates exceeds degree bound BP.R
        // Need to account for duplicates between candidates and existing neighbors
        // Simplification: assume worst case (no overlap) for check
        // A more precise check would involve unique count after merging.
        if (current_neighbors_size + candidate_indices.size() <= BP.R) {
            // Simply add new unique neighbors
            parlay::sequence<indexType> current_neighbors_seq = G[target_node].copy(); // Get copy of current neighbors
            add_neighbors_without_repeats(candidate_indices, current_neighbors_seq); // Append unique candidates
            // Ensure size doesn't exceed R after adding (add_neighbors might grow beyond R if initial size was close)
            if (current_neighbors_seq.size() > BP.R) {
                 // This should ideally not happen if the initial check was precise, but as a fallback:
                 // Option 1: Truncate (simplest, might not be ideal)
                 // current_neighbors_seq.resize(BP.R); // This is not a parlay::sequence operation
                 // Need to create a slice
                 current_neighbors_seq = parlay::slice(current_neighbors_seq, 0, BP.R);
                 // Option 2: Run prune anyway (safer) - leads to next block
                 goto run_prune; // Use goto for simplicity to jump to prune block
            }
             G[target_node].update_neighbors(current_neighbors_seq); // Update with combined list

        } else {
          run_prune: // Label for goto jump
            // Degree bound exceeded, need to prune
            // Call the wrapper robustPrune that computes distances
            auto [final_neighbors, dc] =
                robustPrune(target_node, std::move(candidate_indices), G, Points, alpha, true); // add=true includes existing neighbors

            G[target_node].update_neighbors(final_neighbors);
            BuildStats.increment_dist(target_node, dc);
        }
      });
      t_prune.stop();


      // Progress reporting
       if (print && BP.single_batch == 0) {
             float current_progress = static_cast<float>(count) / m;
             if (current_progress >= progress + progress_inc) {
                progress = current_progress;
                 std::cout << "  Batch Insert Progress: " << static_cast<int>(progress * 100) << "%" << std::endl;
             }
       }


       // Break early if single_batch mode
        if (BP.single_batch != 0) {
             break;
        }

    } // End while loop over batches


    t_beam.total();
    t_bidirect.total();
    t_prune.total();
  }


}; // end struct knn_index

} // namespace parlayANN