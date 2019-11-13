// (c) Yasuhiro Fujii <y-fujii@mimosa-pudica.net>, under MIT License.
#include <random>
#include <Eigen/Dense>

namespace topt {

template<int N>
using Vector = Eigen::Matrix<double, N, 1>;


template<class Functor>
double golden_section_search(double x0, double x3, Functor f) {
	double const s = (sqrt(5.0) - 1.0) / 2.0;
	double x1 = x3 - s * (x3 - x0);
	double x2 = x0 + s * (x3 - x0);
	double f1 = f(x1);
	double f2 = f(x2);
	while (x1 < x2) {
		if (f1 < f2) {
			x3 = x2;
			x2 = x1;
			x1 = x3 - s * (x3 - x0);
			f2 = f1;
			f1 = f(x1);
		}
		else {
			x0 = x1;
			x1 = x2;
			x2 = x0 + s * (x3 - x0);
			f1 = f2;
			f2 = f(x2);
		}
	}
	return (x1 + x2) / 2.0;
}

// ref. <http://www.scholarpedia.org/article/Nelder-Mead_algorithm>.
template<int N, class Functor>
Vector<N> nelder_mead(Vector<N> const& x_min, Vector<N> const& x_max, Vector<N> const& x_tol, double f_tol, Functor f) {
	struct Vertex {
		Vector<N> x;
		double f;
	};

	std::array<Vertex, N + 1> vs;
	for (size_t i = 0; i < N; ++i) {
		vs[i].x = x_min;
		vs[i].x(i) = x_max(i);
		vs[i].f = f(vs[i].x);
	}
	vs[N].x = x_min;
	vs[N].f = f(vs[N].x);

	while (true) {
		size_t best = 0;
		size_t worst = 1;
		size_t worse = 2;
		for (size_t j = 0; j <= N; ++j) {
			if (vs[j].f < vs[best].f) {
				best = j;
			}
			if (vs[j].f > vs[worst].f) {
				worse = worst;
				worst = j;
				assert(worse != worst);
			}
			else if (j != worst && vs[j].f > vs[worse].f) {
				worse = j;
				assert(worse != worst);
			}
		}
		// the condition must be satisfied even if vs[best].f == vs[worse].f == vs[worst].f.
		assert(best != worse && worse != worst && worst != best);

		if (((vs[worst].x - vs[best].x).array().abs() <= x_tol.array()).all() && vs[worst].f - vs[best].f <= f_tol) {
			return vs[best].x;
		}

		// XXX: O(N^2).
		Vector<N> c = Vector<N>::Zero();
		for (size_t j = 0; j <= N; ++j) {
			if (j == worst) {
				continue;
			}
			c += vs[j].x;
		}
		c *= 1.0 / N;

		// reflection.
		Vertex vr;
		vr.x = 2.0 * c - vs[worst].x;
		vr.f = f(vr.x);
		if (vr.f < vs[best].f) {
			// expansion.
			Vertex ve;
			ve.x = 2.0 * vr.x - c;
			ve.f = f(ve.x);
			vs[worst] = ve.f < vr.f ? ve : vr;
		}
		else if (vr.f < vs[worse].f) {
			vs[worst] = vr;
		}
		else {
			// contraction.
			Vertex const& vm = vr.f < vs[worst].f ? vr : vs[worst];
			Vertex vc;
			vc.x = 0.5 * (vm.x + c);
			vc.f = f(vc.x);
			if (vc.f < vm.f) {
				vs[worst] = vc;
			}
			else {
				// shrinkage. O(N^2).
				for (size_t j = 0; j <= N; ++j) {
					if (j == best) {
						continue;
					}
					vs[j].x = 0.5 * (vs[j].x + vs[best].x);
					vs[j].f = f(vs[j].x);
				}
			}
		}
	}
}

template<int N, class Rng, class Functor>
Vector<N> differential_evolution(Vector<N> const& x_min, Vector<N> const& x_max, Vector<N> const& x_tol, double f_tol, Rng& rng, Functor f) {
	struct Vertex {
		Vector<N> x;
		double f;
	};

	std::uniform_real_distribution<double> dist_u01(0.0, 1.0);
	std::array<Vertex, 12 * N> vs;
	for (Vertex& v: vs) {
		auto r = Vector<N>::NullaryExpr([&]{ return dist_u01(rng); });
		v.x = x_min + (x_max - x_min).cwiseProduct(r);
		v.f = f(v.x);
	}

	size_t best  = 0;
	size_t worst = vs.size() - 1;
	for (size_t j = 0; j < vs.size(); ++j) {
		if (vs[j].f < vs[best].f) {
			best = j;
		}
		if (vs[j].f > vs[worst].f) {
			worst = j;
		}
	}
	assert(vs.size() == 1 || best != worst);

	std::uniform_int_distribution<size_t> dist_ia(0, vs.size() - 2);
	std::uniform_int_distribution<size_t> dist_ib(0, vs.size() - 3);
	std::uniform_int_distribution<size_t> dist_ic(0, vs.size() - 4);
	std::uniform_int_distribution<size_t> dist_ix(0, N - 1);
	while (((vs[worst].x - vs[best].x).array().abs() > x_tol.array()).any() || vs[worst].f - vs[best].f > f_tol) {
		for (size_t j = 0; j < vs.size(); ++j) {
			size_t ia = dist_ia(rng);
			if (ia == j) {
				ia = vs.size() - 1;
			}
			size_t ib = dist_ib(rng);
			if (ib == j) {
				ib = vs.size() - 1;
			}
			if (ib == ia) {
				ib = vs.size() - 2;
			}
			size_t ic = dist_ic(rng);
			if (ic == j) {
				ic = vs.size() - 1;
			}
			if (ic == ia) {
				ic = vs.size() - 2;
			}
			if (ic == ib) {
				ic = vs.size() - 3;
			}
			size_t ix = dist_ix(rng);
			Vertex v;
			for (size_t k = 0; k < N; ++k) {
				if (dist_u01(rng) < 0.5 || k == ix) {
					v.x(k) = vs[ia].x(k) + 0.5 * (vs[ib].x(k) - vs[ic].x(k));
				}
				else {
					v.x(k) = vs[j].x(k);
				}
			}
			v.f = f(v.x);
			if (v.f < vs[j].f) {
				vs[j] = v;
				if (v.f < vs[best].f) {
					best = j;
				}
				if (j == worst) {
					// XXX: O(#vs).
					for (size_t k = 0; k < vs.size(); ++k) {
						if (vs[k].f > vs[worst].f) {
							worst = k;
						}
					}
				}
			}
		}
	}

	return vs[best].x;
}


template<int N, class Rng, class Functor>
Vector<N> evolution_strategy_simple(Vector<N> x, double sigma, double sigma_tol, Rng& rng, Functor f) {
	double m_sigma_accept = std::exp(+1.0 / (1.0 * 256.0 * N));
	double m_sigma_reject = std::exp(-1.0 / (4.0 * 256.0 * N));

	std::normal_distribution<double> dist_n01;
	double fx = f(x);
	while (sigma > sigma_tol) {
		auto n01 = Vector<N>::NullaryExpr([&]{ return dist_n01(rng); });
		Vector<N> y = x + sigma * n01;
		double fy = f(y);
		if (fy < fx) {
			x = y;
			fx = fy;
			sigma *= m_sigma_accept;
		}
		else {
			sigma *= m_sigma_reject;
		}
	}
	return x;
}


}
