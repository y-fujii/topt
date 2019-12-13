// (c) Yasuhiro Fujii <y-fujii@mimosa-pudica.net>, under MIT License.
#include <random>
#include <vector>
#include <Eigen/Dense>

namespace topt {


template<int N>
using Vector = Eigen::Matrix<double, N, 1>;

template<int N>
struct Vertex {
	Vector<N> x;
	double f;
};

template<int N, class Derived>
void kahan_add(Vector<N>& x0, Vector<N>& x1, Eigen::MatrixBase<Derived> const& y0) {
	Vector<N> y = x1 + y0;
	Vector<N> x = x0 + y;
	x1 = y - (x - x0);
	x0 = x;
}

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
Vertex<N> nelder_mead(Vector<N> const& x_min, Vector<N> const& x_max, Vector<N> const& x_tol, double const f_tol, Functor f) {
	static_assert(N >= 2);
	constexpr bool on_algorithm = N > 8; // XXX: needs benchmarks.

	std::vector<Vertex<N>, Eigen::aligned_allocator<Vertex<N>>> vs(1 + N);
	for (size_t i = 0; i < N; ++i) {
		vs[i].x = x_min;
		vs[i].x(i) = x_max(i);
		vs[i].f = f(vs[i].x);
	}
	vs[N].x = x_min;
	vs[N].f = f(vs[N].x);

	Vector<N> s0 = Vector<N>::Zero();
	Vector<N> s1 = Vector<N>::Zero(); (void)s1;
	if constexpr (on_algorithm) {
		for (size_t j = 0; j < 1 + N; ++j) {
			kahan_add(s0, s1, vs[j].x);
		}
	}

	while (true) {
		// XXX:
		// vs[0].f = 1.0;
		// vs[1].f = 0.0;
		// vs[2].f = 0.0;
		size_t best = 0;
		size_t worst = 1;
		size_t worse = 2;
		for (size_t j = 0; j < 1 + N; ++j) {
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
		//assert(best != worse && worse != worst && worst != best);

		if (((vs[worst].x - vs[best].x).array().abs() <= x_tol.array()).all() && vs[worst].f - vs[best].f <= f_tol) {
			return vs[best];
		}

		if constexpr (on_algorithm) {
			kahan_add(s0, s1, -vs[worst].x);
		}
		else {
			s0 = Vector<N>::Zero();
			for (size_t j = 0; j < 1 + N; ++j) {
				if (j == worst) {
					continue;
				}
				s0 += vs[j].x;
			}
		}

		Vector<N> c = (1.0 / N) * s0;

		// reflection.
		Vertex<N> vr;
		vr.x = 2.0 * c - vs[worst].x;
		vr.f = f(vr.x);
		if (vr.f < vs[best].f) {
			// expansion.
			Vertex<N> ve;
			ve.x = 2.0 * vr.x - c;
			ve.f = f(ve.x);
			vs[worst] = ve.f < vr.f ? ve : vr;
		}
		else if (vr.f < vs[worse].f) {
			vs[worst] = vr;
		}
		else {
			// contraction.
			Vertex<N> const& vm = vr.f < vs[worst].f ? vr : vs[worst];
			Vertex<N> vc;
			vc.x = 0.5 * (vm.x + c);
			vc.f = f(vc.x);
			if (vc.f < vm.f) {
				vs[worst] = vc;
			}
			else {
				if constexpr (on_algorithm) {
					s0 = Vector<N>::Zero();
					s1 = Vector<N>::Zero();
				}
				// shrinkage. O(N^2).
				for (size_t j = 0; j < 1 + N; ++j) {
					if (j != best) {
						vs[j].x = 0.5 * (vs[j].x + vs[best].x);
						vs[j].f = f(vs[j].x);
					}
					if constexpr (on_algorithm) {
						kahan_add(s0, s1, vs[j].x);
					}
				}
				continue;
			}
		}

		if constexpr (on_algorithm) {
			kahan_add(s0, s1, vs[worst].x);
		}
	}
}

template<int N, class Rng, class Functor>
Vertex<N> differential_evolution(Vector<N> const& x_min, Vector<N> const& x_max, Vector<N> const& x_tol, double const f_tol, Rng& rng, Functor f) {
	static_assert(N >= 1);

	std::uniform_real_distribution<double> dist_u01(0.0, 1.0);
	std::vector<Vertex<N>, Eigen::aligned_allocator<Vertex<N>>> vs(12 * N);
	for (Vertex<N>& v: vs) {
		auto const r = Vector<N>::NullaryExpr([&]{ return dist_u01(rng); });
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
	assert(best != worst);

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
			size_t const ix = dist_ix(rng);
			Vertex<N> v;
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

	return vs[best];
}


template<int N, class Rng, class Functor>
Vertex<N> evolution_strategy_simple(Vector<N> const& x, double sigma, double const sigma_tol, Rng& rng, Functor f) {
	double const m_sigma_accept = std::exp(+1.0 / (1.0 * 64.0 * N));
	double const m_sigma_reject = std::exp(-1.0 / (4.0 * 64.0 * N));

	std::normal_distribution<double> dist_n01;
	Vertex<N> v = {x, f(x)};
	while (sigma > sigma_tol) {
		auto const n01 = Vector<N>::NullaryExpr([&]{ return dist_n01(rng); });
		Vertex<N> w;
		w.x = v.x + sigma * n01;
		w.f = f(w.x);
		if (w.f < v.f) {
			v = w;
			sigma *= m_sigma_accept;
		}
		else {
			sigma *= m_sigma_reject;
		}
	}
	return v;
}

template<int N, class Rng, class Functor>
Vertex<N> simulated_annealing(Vector<N> const& x, double sigma, double sigma_tol, double const k, Rng& rng, Functor f) {
	double const m_beta = std::exp(1.0 / (k * N));
	double const m_sigma_accept = std::exp(+1.0 / (1.0 * 64.0 * N));
	double const m_sigma_reject = std::exp(-1.0 / (4.0 * 64.0 * N));

	std::normal_distribution<double> dist_n01;
	double beta = 1.0;
	Vertex<N> v = {x, f(x)};
	while (sigma > sigma_tol) {
		auto const n01 = Vector<N>::NullaryExpr([&]{ return dist_n01(rng); });
		Vertex<N> w;
		w.x = v.x + sigma * n01;
		w.f = f(w.x);
		if (rng() < rng.max() * std::exp(beta * (v.f - w.f))) {
			v = w;
			sigma *= m_sigma_accept;
		}
		else {
			sigma *= m_sigma_reject;
		}
		beta *= m_beta;
	}
	return v;
}

// TODO: CMA-ES.
// ref. <https://arxiv.org/pdf/1604.00772.pdf>.


}
