/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <cmath>

namespace TensileLite
{
    namespace Matching
    {
        /**
 * \ingroup PropertyMatching
 * \defgroup DistanceFunctions Distance Functions
 */

        /**
 * \addtogroup DistanceFunctions
 * @{
 */

        /**
 * @brief Abstract Distance function base class
 */

        /**
 * Distance functions must implement the following methods:
 *
 *      double operator()(Key const& p1, Key const& p2)
 *
 * Returns the distance between p1 and p2.
 *
 *      bool improvementPossible(Key const& p1, Key const& p2, size_t idx,
 * double bestDistance) const
 *
 * May return `false` if it's impossible for `p1` and `p2` to be closer than
 * `bestDistance` based on `p1[idx]` and `p2[idx]` alone. This can be used to
 * exit early when searching a table sorted by element `idx`.
 *
 * If that doesn't apply to the given distance metric, returning `true` will
 * give the correct result.
 */
        template <typename Key>
        class Distance
        {
        public:
            virtual std::string type() const = 0;
            virtual ~Distance()              = default;
        };

        // Not really a distance, but defined as one for template specialization purposes
        template <typename Key>
        struct Equality : public Distance<Key>
        {
            enum
            {
                HasIndex = false,
                HasValue = false
            };
            static std::string Type()
            {
                return "Equality";
            }
            virtual std::string type() const override
            {
                return Type();
            }

            inline bool operator()(Key const& p1, Key const& p2) const
            {
                double distance = 0.0;

                for(int i = 0; i < p1.size(); i++)
                {
                    double di = p1[i] - p2[i];
                    distance += di * di;
                }
                return distance;
            }
        };

        // Not really a distance, but defined as one for template specialization purposes
        template <typename Key>
        struct Range : public Distance<Key>
        {
            enum
            {
                HasIndex = false,
                HasValue = false
            };
            static std::string Type()
            {
                return "Range";
            }
            virtual std::string type() const override
            {
                return Type();
            }

            inline bool operator()(Key const& p1, Key const& p2) const
            {
                double distance = 0.0;

                for(int i = 0; i < p1.size(); i++)
                {
                    double di = p1[i] - p2[i];
                    distance += di * di;
                }
                return distance;
            }
        };

        template <typename Key>
        struct RatioDistance : public Distance<Key>
        {
            enum
            {
                HasIndex = false,
                HasValue = false
            };
            static std::string Type()
            {
                return "Ratio";
            }
            virtual std::string type() const override
            {
                return Type();
            }

            inline double operator()(Key const& p1, Key const& p2) const
            {
                double distance = 1.0;
                for(int i = 0; i < p1.size(); i++)
                {
                    distance += std::abs(std::log(double(p1[i]) / double(p2[i])));
                }

                return distance;
            }

            inline bool improvementPossible(Key const& p1,
                                            Key const& p2,
                                            size_t     idx,
                                            double     bestDistance) const
            {
                return true;
            }
        };

        template <typename Key>
        struct ManhattanDistance : public Distance<Key>
        {
            enum
            {
                HasIndex = false,
                HasValue = false
            };
            static std::string Type()
            {
                return "Manhattan";
            }
            virtual std::string type() const override
            {
                return Type();
            }

            inline double operator()(Key const& p1, Key const& p2) const
            {
                double distance = 0;
                for(int i = 0; i < p1.size(); i++)
                {
                    distance += std::abs(p1[i] - p2[i]);
                }
                return distance;
            }

            inline bool improvementPossible(Key const& p1,
                                            Key const& p2,
                                            size_t     idx,
                                            double     bestDistance) const
            {
                double d0 = std::abs(p1[idx] - p2[idx]);
                return (d0 < bestDistance) || (p1 == p2);
            }
        };

        template <typename Key>
        struct EuclideanDistance : public Distance<Key>
        {
            enum
            {
                HasIndex = false,
                HasValue = false
            };

            static std::string Type()
            {
                return "Euclidean";
            }
            virtual std::string type() const override
            {
                return Type();
            }

            inline double operator()(Key const& p1, Key const& p2) const
            {
                double distance = 0.0;

                for(int i = 0; i < p1.size(); i++)
                {
                    double di = p1[i] - p2[i];
                    distance += di * di;
                }
                return distance;
            }

            inline bool improvementPossible(Key const& p1,
                                            Key const& p2,
                                            size_t     idx,
                                            double     bestDistance) const
            {
                double d0 = p1[idx] - p2[idx];
                return ((d0 * d0) < bestDistance) || (p1 == p2);
            }
        };

        template <typename Key>
        struct JSDivergence : public Distance<Key>
        {
            enum
            {
                HasIndex = false,
                HasValue = false
            };

            static std::string Type()
            {
                return "JSD";
            }
            virtual std::string type() const override
            {
                return Type();
            }

            double kl_divergence(double* p, double* q) const
            {
                double acc  = 0.0;
                int    dims = 3;

                for(int i = 0; i < dims; i++)
                    acc += p[i] * std::log(p[i] / q[i]);

                return acc;
            }

            void normalize(Key const& p, Key const& q, double* norm_p, double* norm_q) const
            {
                int const dims  = 3;
                double    sum_p = 0.0;
                double    sum_q = 0.0;

                for(int i = 0; i < dims; i++)
                {
                    sum_p += p[i];
                    sum_q += q[i];
                }

                for(int i = 0; i < dims; i++)
                {
                    norm_p[i] = p[i] / sum_p;
                    norm_q[i] = q[i] / sum_q;
                }
            }

            inline double operator()(Key const& p1, Key const& p2) const
            {
                double distance = 0.0;

                double norm_p[3];
                double norm_q[3];
                double m[3];

                this->normalize(p1, p2, norm_p, norm_q);

                for(int i = 0; i < p1.size(); i++)
                    m[i] = 0.5 * (norm_p[i] + norm_q[i]);

                distance
                    = 0.5 * this->kl_divergence(norm_p, m) + 0.5 * this->kl_divergence(norm_q, m);

                return distance;
            }

            inline bool improvementPossible(Key const& p1,
                                            Key const& p2,
                                            size_t     idx,
                                            double     bestDistance) const
            {
                double norm_p[3];
                double norm_q[3];
                double m;

                this->normalize(p1, p2, norm_p, norm_q);

                m = 0.5 * (norm_p[idx] + norm_q[idx]);

                double d0 = 0.5 * (norm_p[idx] * std::log(norm_p[idx] / m))
                            + 0.5 * (norm_q[idx] * std::log(norm_q[idx] / m));

                return (d0 < bestDistance) || (p1 == p2);
            }
        };

        template <typename Key>
        struct RandomDistance : public Distance<Key>
        {
            enum
            {
                HasIndex = false,
                HasValue = false
            };

            static std::string Type()
            {
                return "Random";
            }
            virtual std::string type() const override
            {
                return Type();
            }

            inline double operator()(Key const& p1, Key const& p2) const
            {
                return double(rand());
            }

            inline bool improvementPossible(Key const& p1,
                                            Key const& p2,
                                            size_t     idx,
                                            double     bestDistance) const
            {
                return true;
            }
        };

        template <typename Key>
        struct GridBasedDistance : public Distance<Key>
        {
            enum
            {
                HasIndex = false,
                HasValue = false
            };

            static std::string Type()
            {
                return "GridBased";
            }
            virtual std::string type() const override
            {
                return Type();
            }

            inline double operator()(Key const& p1, Key const& p2) const
            {
                double distance = 0.0;
                double gridK = p2.size() > 3 ? p2[3] : p2[2];

                // and nearest K
                double K = p1.size() > 3 ? p1[3] : p1[2];
                distance = abs(K - gridK);

                return distance;
            }

            inline bool improvementPossible(Key const& p1,
                                            Key const& p2,
                                            size_t     idx,
                                            double     bestDistance) const
            {
                return true;
            }
        };
        /**
 * @}
 */

    } // namespace Matching
} // namespace TensileLite
