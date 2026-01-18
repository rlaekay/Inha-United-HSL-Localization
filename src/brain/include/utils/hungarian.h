#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

using namespace std;

class HungarianAlgorithm {
public:
  HungarianAlgorithm() {}
  ~HungarianAlgorithm() {}

  /**
   * @brief Solve the assignment problem
   * @param DistMatrix Input cost matrix (rows * cols)
   * @param Assignment Output vector containing the column index for each row. -1 if unassigned.
   * @return Minimum total cost
   */
  double Solve(const vector<vector<double>> &DistMatrix, vector<int> &Assignment) {
    int nRows = DistMatrix.size();
    if (nRows == 0) return 0.0;
    int nCols = DistMatrix[0].size();
    if (nCols == 0) return 0.0;

    double cost = 0.0;
    Assignment.assign(nRows, -1);

    int n = max(nRows, nCols);

    // Ensure buffers are large enough
    int sizeNeeded = n + 1;
    if (u.size() < sizeNeeded) {
      u.resize(sizeNeeded);
      v.resize(sizeNeeded);
      p.resize(sizeNeeded);
      way.resize(sizeNeeded);
      minv.resize(sizeNeeded);
      used.resize(sizeNeeded);
      // Resize cost matrix rows
      costMatrix.resize(n);
      for (auto &row : costMatrix)
        row.resize(n);
    }
    // Even if size is enough, make sure costMatrix cols are enough (if n grew)
    if (costMatrix.size() < n) { costMatrix.resize(n); }
    for (int i = 0; i < n; ++i) {
      if (costMatrix[i].size() < n) costMatrix[i].resize(n);
    }

    // Fill cost matrix
    // Initialize with 0 for padding (or strict requirement to match closest, 0 works for min cost general)
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        if (i < nRows && j < nCols) {
          costMatrix[i][j] = DistMatrix[i][j];
        } else {
          costMatrix[i][j] = 0.0;
        }
      }
    }

    // Reset helper arrays (fill not strictly needed if we init correctly in loop, but safer)
    fill(u.begin(), u.end(), 0.0);
    fill(v.begin(), v.end(), 0.0);
    fill(p.begin(), p.end(), 0);
    fill(way.begin(), way.end(), 0);

    // --- Hungarian Algorithm (O(n^3)) ---

    for (int i = 1; i <= n; ++i) {
      p[0] = i;
      int j0 = 0;
      // Reset minv and used for this iteration step
      // We only strictly need to reset up to n
      fill(minv.begin(), minv.begin() + sizeNeeded, std::numeric_limits<double>::infinity());
      fill(used.begin(), used.begin() + sizeNeeded, false);

      do {
        used[j0] = true;
        int i0 = p[j0], j1 = 0;
        double delta = std::numeric_limits<double>::infinity();

        for (int j = 1; j <= n; ++j) {
          if (!used[j]) {
            double cur = costMatrix[i0 - 1][j - 1] - u[i0] - v[j];
            if (cur < minv[j]) {
              minv[j] = cur;
              way[j] = j0;
            }
            if (minv[j] < delta) {
              delta = minv[j];
              j1 = j;
            }
          }
        }

        for (int j = 0; j <= n; ++j) {
          if (used[j]) {
            u[p[j]] += delta;
            v[j] -= delta;
          } else {
            minv[j] -= delta;
          }
        }
        j0 = j1;
      } while (p[j0] != 0);

      do {
        int j1 = way[j0];
        p[j0] = p[j1];
        j0 = j1;
      } while (j0 != 0);
    }

    for (int j = 1; j <= n; ++j) {
      if (p[j] != 0) {
        int row = p[j] - 1;
        int col = j - 1;
        if (row < nRows && col < nCols) {
          Assignment[row] = col;
          cost += DistMatrix[row][col];
        }
      }
    }

    return cost;
  }

private:
  // Persistent buffers to avoid allocation
  vector<double> u, v, minv;
  vector<int> p, way;
  vector<bool> used;
  vector<vector<double>> costMatrix;
};
