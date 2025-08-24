// fastcarbs_core.cpp

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <random>
#include <stdexcept>

namespace py = pybind11;
using Eigen::MatrixXd;
using Eigen::VectorXd;

static inline double softplus(double x) { // unused, we parameterize with exp()
    if (x > 20.0) return x; // avoid overflow
    if (x < -20.0) return std::exp(x);
    return std::log1p(std::exp(x));
}

struct Adam {
    double lr, b1, b2, eps;
    std::vector<double> m, v;
    int t = 0;
    Adam(size_t n, double lr_=1e-4, double b1_=0.9, double b2_=0.999, double eps_=1e-8)
        : lr(lr_), b1(b1_), b2(b2_), eps(eps_), m(n, 0.0), v(n, 0.0) {}
    void step(std::vector<double>& params, const std::vector<double>& grad) {
        t++;
        double b1t = std::pow(b1, t);
        double b2t = std::pow(b2, t);
        double alpha = lr * std::sqrt(1.0 - b2t) / (1.0 - b1t);
        for (size_t i = 0; i < params.size(); ++i) {
            m[i] = b1*m[i] + (1.0 - b1)*grad[i];
            v[i] = b2*v[i] + (1.0 - b2)*grad[i]*grad[i];
            params[i] -= alpha * m[i] / (std::sqrt(v[i]) + eps);
        }
    }
};

struct Kernel {
    virtual ~Kernel() = default;
    virtual void K(const MatrixXd& X, MatrixXd& Kout) const = 0;
    virtual void K_cross(const MatrixXd& Xa, const MatrixXd& Xb, MatrixXd& Kout) const = 0;
    virtual double k(const Eigen::Ref<const Eigen::VectorXd>& xa,
                     const Eigen::Ref<const Eigen::VectorXd>& xb) const = 0;
    virtual void grad_params(const MatrixXd& X, const MatrixXd& A, std::vector<double>& g) const = 0;
    virtual size_t num_params() const = 0;
    virtual void set_params(const std::vector<double>& p) = 0;  // unconstrained
    virtual void get_params(std::vector<double>& p) const = 0;  // unconstrained
};

// Matern-3/2 with ARD: k = sig2 * (1 + a r) exp(-a r), r = sqrt(sum_j (d_j^2 / l_j^2)), a = sqrt(3)
struct Matern32ARD : public Kernel {
    // unconstrained parameters are logs: log(sig2), log(l_j) for each dim
    double log_sig2;                  // amplitude variance
    std::vector<double> log_ls;       // ARD lengthscales
    const double a = std::sqrt(3.0);

    explicit Matern32ARD(size_t d, double init_ls, double init_sig2=1.0)
      : log_sig2(std::log(init_sig2)), log_ls(d, std::log(init_ls)) {}

    size_t D() const { return log_ls.size(); }

    inline double sig2() const { return std::exp(log_sig2); }

    inline void inv_ls_sq(std::vector<double>& invls2) const {
        invls2.resize(D());
        for (size_t i=0;i<D();++i) {
            double li = std::exp(log_ls[i]);
            invls2[i] = 1.0/(li*li);
        }
    }

    double k(const Eigen::Ref<const VectorXd>& xa,
             const Eigen::Ref<const VectorXd>& xb) const override {
        std::vector<double> invls2;
        inv_ls_sq(invls2);
        double s = 0.0;
        for (size_t j=0;j<D();++j) {
            double d = xa[j]-xb[j];
            s += d*d*invls2[j];
        }
        double r = std::sqrt(std::max(0.0, s));
        double term = (1.0 + a*r)*std::exp(-a*r);
        return sig2() * term;
    }

    void K(const MatrixXd& X, MatrixXd& Kout) const override {
        const int n = X.rows();
        Kout.resize(n, n);
        std::vector<double> invls2;
        inv_ls_sq(invls2);
        for (int i=0;i<n;++i) {
            Kout(i,i) = sig2();
            for (int j=i+1;j<n;++j) {
                double s = 0.0;
                for (int k=0;k<(int)D();++k) {
                    double d = X(i,k)-X(j,k);
                    s += d*d*invls2[k];
                }
                double r = std::sqrt(std::max(0.0, s));
                double term = (1.0 + a*r)*std::exp(-a*r);
                double val = sig2()*term;
                Kout(i,j)=val;
                Kout(j,i)=val;
            }
        }
    }

    void K_cross(const MatrixXd& Xa, const MatrixXd& Xb, MatrixXd& Kout) const override {
        const int na = Xa.rows(), nb = Xb.rows();
        Kout.resize(na, nb);
        std::vector<double> invls2;
        inv_ls_sq(invls2);
        for (int i=0;i<na;++i) {
            for (int j=0;j<nb;++j) {
                double s = 0.0;
                for (int k=0;k<(int)D();++k) {
                    double d = Xa(i,k)-Xb(j,k);
                    s += d*d*invls2[k];
                }
                double r = std::sqrt(std::max(0.0, s));
                double term = (1.0 + a*r)*std::exp(-a*r);
                Kout(i,j) = sig2()*term;
            }
        }
    }

    void grad_params(const MatrixXd& X, const MatrixXd& A, std::vector<double>& g) const override {
        const int n = X.rows();
        g.assign(1 + D(), 0.0);
        std::vector<double> invls2;
        inv_ls_sq(invls2);
        const double sig2v = sig2();

        double g_sig2 = 0.0;
        for (int i=0;i<n;++i) {
            g_sig2 += 0.5 * A(i,i);
            for (int j=i+1;j<n;++j) {
                double s = 0.0;
                for (size_t k=0;k<D();++k) {
                    double d = X(i,k)-X(j,k);
                    s += d*d*invls2[k];
                }
                double r = std::sqrt(std::max(0.0, s));
                double term = (1.0 + a*r)*std::exp(-a*r);
                double contrib = (A(i,j)+A(j,i)) * term * 0.5;
                g_sig2 += contrib;
            }
        }
        g[0] = g_sig2 * sig2v;

        for (size_t dim=0; dim<D(); ++dim) {
            double li = std::exp(log_ls[dim]);
            double invli = 1.0/li;
            double invli3 = invli*invli*invli;
            double accum = 0.0;
            for (int i=0;i<n;++i) {
                for (int j=i+1;j<n;++j) {
                    double s = 0.0;
                    for (size_t k=0;k<D();++k) {
                        double d = X(i,k)-X(j,k);
                        double invlk = 1.0/std::exp(log_ls[k]);
                        s += d*d*invlk*invlk;
                    }
                    double r = std::sqrt(std::max(0.0, s));
                    double d_dim = X(i,dim)-X(j,dim);
                    double base = std::exp(-a*r);
                    double dk_dli = sig2v * a*a * base * (d_dim*d_dim) * invli3;
                    double contrib = (A(i,j)+A(j,i)) * dk_dli * 0.5;
                    accum += contrib;
                }
            }
            g[1+dim] = accum * li;
        }
    }

    size_t num_params() const override { return 1 + D(); }

    void set_params(const std::vector<double>& p) override {
        if (p.size() != num_params()) throw std::runtime_error("Matern32ARD bad param size");
        log_sig2 = p[0];
        for (size_t i=0;i<D();++i) log_ls[i] = p[1+i];
    }
    void get_params(std::vector<double>& p) const override {
        p.resize(num_params());
        p[0] = log_sig2;
        for (size_t i=0;i<D();++i) p[1+i] = log_ls[i];
    }
};

// Linear kernel
struct LinearKernel : public Kernel {
    double log_sig2;
    size_t d;
    explicit LinearKernel(size_t d_, double init_sig2=1.0) : log_sig2(std::log(init_sig2)), d(d_) {}

    inline double sig2() const { return std::exp(log_sig2); }

    double k(const Eigen::Ref<const VectorXd>& xa,
             const Eigen::Ref<const VectorXd>& xb) const override { return sig2() * xa.dot(xb); }
    void K(const MatrixXd& X, MatrixXd& Kout) const override { Kout = sig2() * (X * X.transpose()); }
    void K_cross(const MatrixXd& Xa, const MatrixXd& Xb, MatrixXd& Kout) const override {
        Kout = sig2() * (Xa * Xb.transpose());
    }

    void grad_params(const MatrixXd& X, const MatrixXd& A, std::vector<double>& g) const override {
        // dK/dsig2 = X X^T  -> gradient = 0.5 trace(A * dK/dsig2) * chain(log)
        MatrixXd XXt = X * X.transpose();
        double tr = (A.cwiseProduct(XXt)).sum();
        g.assign(1, 0.5 * tr * sig2());
    }

    size_t num_params() const override { return 1; }
    void set_params(const std::vector<double>& p) override {
        if (p.size()!=1) throw std::runtime_error("LinearKernel bad param size");
        log_sig2 = p[0];
    }
    void get_params(std::vector<double>& p) const override { p = {log_sig2}; }
};

// Sum kernel
struct SumKernel : public Kernel {
    std::unique_ptr<Kernel> k1, k2;
    SumKernel(std::unique_ptr<Kernel> a, std::unique_ptr<Kernel> b)
      : k1(std::move(a)), k2(std::move(b)) {}
    double k(const Eigen::Ref<const VectorXd>& xa,
             const Eigen::Ref<const VectorXd>& xb) const override {
        return k1->k(xa, xb) + k2->k(xa, xb);
    }
    void K(const MatrixXd& X, MatrixXd& Kout) const override {
        MatrixXd A, B; k1->K(X, A); k2->K(X, B); Kout = A + B;
    }
    void K_cross(const MatrixXd& Xa, const MatrixXd& Xb, MatrixXd& Kout) const override {
        MatrixXd A, B; k1->K_cross(Xa, Xb, A); k2->K_cross(Xa, Xb, B); Kout = A + B;
    }
    void grad_params(const MatrixXd& X, const MatrixXd& A, std::vector<double>& g) const override {
        std::vector<double> g1, g2; k1->grad_params(X, A, g1); k2->grad_params(X, A, g2);
        g.clear(); g.reserve(g1.size()+g2.size()); g.insert(g.end(), g1.begin(), g1.end()); g.insert(g.end(), g2.begin(), g2.end());
    }
    size_t num_params() const override { return k1->num_params() + k2->num_params(); }
    void set_params(const std::vector<double>& p) override {
        size_t n1 = k1->num_params();
        std::vector<double> p1(p.begin(), p.begin()+n1), p2(p.begin()+n1, p.end());
        k1->set_params(p1); k2->set_params(p2);
    }
    void get_params(std::vector<double>& p) const override {
        std::vector<double> p1, p2; k1->get_params(p1); k2->get_params(p2);
        p.clear(); p.reserve(p1.size()+p2.size()); p.insert(p.end(), p1.begin(), p1.end()); p.insert(p.end(), p2.begin(), p2.end());
    }
};

// RBF (1D) kernel
struct RBF1D : public Kernel {
    double log_sig2;
    double log_l;
    explicit RBF1D(double init_l=1.0, double init_sig2=1.0)
      : log_sig2(std::log(init_sig2)), log_l(std::log(init_l)) {}
    inline double sig2() const { return std::exp(log_sig2); }
    inline double l() const { return std::exp(log_l); }
    double k(const Eigen::Ref<const VectorXd>& xa,
             const Eigen::Ref<const VectorXd>& xb) const override {
        double d = xa[0] - xb[0];
        double q = (d*d) / (l()*l());
        return sig2() * std::exp(-0.5*q);
    }
    void K(const MatrixXd& X, MatrixXd& Kout) const override {
        int n = X.rows();
        Kout.resize(n,n);
        for (int i=0;i<n;++i) {
            Kout(i,i) = sig2();
            for (int j=i+1;j<n;++j) {
                double d = X(i,0)-X(j,0);
                double q = (d*d)/(l()*l());
                double val = sig2()*std::exp(-0.5*q);
                Kout(i,j)=val; Kout(j,i)=val;
            }
        }
    }
    void K_cross(const MatrixXd& Xa, const MatrixXd& Xb, MatrixXd& Kout) const override {
        int na = Xa.rows(), nb = Xb.rows();
        Kout.resize(na, nb);
        for (int i=0;i<na;++i) for (int j=0;j<nb;++j) {
            double d = Xa(i,0)-Xb(j,0);
            double q = (d*d)/(l()*l());
            Kout(i,j) = sig2()*std::exp(-0.5*q);
        }
    }
    void grad_params(const MatrixXd& X, const MatrixXd& A, std::vector<double>& g) const override {
        int n = X.rows();
        double g_sig2 = 0.0, g_l = 0.0;
        double sig2v = sig2(), lv = l();
        double invl2 = 1.0/(lv*lv);
        for (int i=0;i<n;++i) {
            g_sig2 += 0.5 * A(i,i);  // add the 0.5
            for (int j=i+1;j<n;++j) {
                double d = X(i,0)-X(j,0);
                double q = (d*d)*invl2;
                double e = std::exp(-0.5*q);
                double k = sig2v * e;
                double contrib = (A(i,j)+A(j,i)) * e * 0.5;
                g_sig2 += contrib;
                // ∂k/∂l = k * ( (d^2)/(l^3) )
                double dk_dl = k * ( (d*d) / (lv*lv*lv) );
                double contrib_l = (A(i,j)+A(j,i)) * dk_dl * 0.5;
                g_l += contrib_l;
            }
        }
        g = { g_sig2 * sig2v, g_l * lv }; // d/d logs
    }
    size_t num_params() const override { return 2; }
    void set_params(const std::vector<double>& p) override { log_sig2 = p[0]; log_l = p[1]; }
    void get_params(std::vector<double>& p) const override { p = {log_sig2, log_l}; }
};

class GPRegression {
public:
    std::unique_ptr<Kernel> kernel;
    double log_noise;     // noise variance in log space
    double jitter;        // numeric jitter added to K
    MatrixXd X;
    VectorXd y;

    // Cached after fit:
    MatrixXd K;       // n x n
    Eigen::LLT<MatrixXd> chol; // Cholesky of K
    VectorXd alpha;   // K^{-1} y

    GPRegression(std::unique_ptr<Kernel> k, double noise_var=1e-2, double jitter_=1e-4)
      : kernel(std::move(k)), log_noise(std::log(noise_var)), jitter(jitter_) {}

    void set_data(py::array_t<double, py::array::c_style | py::array::forcecast> Xnp,
                  py::array_t<double, py::array::c_style | py::array::forcecast> ynp) {
        auto Xbuf = Xnp.request(), ybuf = ynp.request();
        const int n = Xbuf.shape[0];
        const int d = Xbuf.shape.size() == 1 ? 1 : Xbuf.shape[1];
        if (ybuf.shape[0] != n) throw std::runtime_error("X,y size mismatch");
        X = MatrixXd(n, d);
        y = VectorXd(n);
        // copy
        const double* Xp = static_cast<const double*>(Xbuf.ptr);
        for (int i=0;i<n;++i) for (int j=0;j<d;++j) X(i,j) = Xp[i*d + j];
        const double* yp = static_cast<const double*>(ybuf.ptr);
        for (int i=0;i<n;++i) y[i] = yp[i];
    }

    double nlml_and_grad(std::vector<double>& g) {
        const int n = X.rows();
        // Build K
        kernel->K(X, K);
        K.diagonal().array() += std::exp(log_noise) + jitter;
        chol.compute(K);
        if (chol.info()!=Eigen::Success) {
            // Add more jitter if needed
            double add = jitter;
            for (int tries=0; tries<6 && chol.info()!=Eigen::Success; ++tries) {
                add *= 10.0;
                K = K + add * MatrixXd::Identity(n,n);
                chol.compute(K);
            }
            if (chol.info()!=Eigen::Success) throw std::runtime_error("Cholesky failed");
        }
        // alpha = K^{-1} y
        alpha = chol.solve(y);
        // NLML = 0.5 y^T alpha + sum(log(diag(L))) + 0.5 n log(2π)
        double term1 = 0.5 * y.dot(alpha);
        double logdet = 0.0;
        for (int i=0;i<n;++i) logdet += std::log(chol.matrixL()(i,i));
        double nlml = term1 + logdet + 0.5 * n * std::log(2.0*M_PI);

        // A = alpha alpha^T - K^{-1}
        MatrixXd Linv = chol.matrixL().solve(MatrixXd::Identity(n,n)); // L^{-1}
        MatrixXd Kinv = Linv.transpose() * Linv;
        MatrixXd A = alpha*alpha.transpose() - Kinv;

        // Grad w.r.t. kernel params
        std::vector<double> gk;
        kernel->grad_params(X, A, gk);

        // Grad w.r.t. noise (variance): dK/dσ_n^2 = I
        double g_noise = 0.5 * (A.cwiseProduct(MatrixXd::Identity(n,n))).sum();
        // d/d log_noise = d/d σ_n^2 * σ_n^2
        g_noise *= std::exp(log_noise);

        // Assemble gradient vector: [kernel params..., log_noise]
        g = gk;
        g.push_back(g_noise);

        return nlml;
    }

    void fit(int num_steps, double lr) {
        // Pack parameters into a flat vector
        std::vector<double> params;
        kernel->get_params(params);
        params.push_back(log_noise); // last entry

        Adam opt(params.size(), lr);
        for (int t=0; t<num_steps; ++t) {
            // Unpack
            std::vector<double> kp(params.begin(), params.end()-1);
            double ln = params.back();
            kernel->set_params(kp);
            log_noise = ln;

            // Forward + grad
            std::vector<double> g;
            double loss = nlml_and_grad(g);

            // Adam step
            opt.step(params, g);
        }
        // Finalize
        std::vector<double> kp(params.begin(), params.end()-1);
        kernel->set_params(kp);
        log_noise = params.back();
        // Cache final K, chol, alpha
        std::vector<double> g;
        (void) nlml_and_grad(g);
    }

    // Predict: returns mean and variance (diagonal), noiseless flag removes noise from variance
    std::pair<py::array_t<double>, py::array_t<double>> predict(
        py::array_t<double, py::array::c_style | py::array::forcecast> Xq_np,
        bool noiseless=true) {

        auto Qbuf = Xq_np.request();
        int nq = Qbuf.shape[0];
        int d = Qbuf.shape.size() == 1 ? 1 : Qbuf.shape[1];
        MatrixXd Xq(nq, d);
        const double* Xp = static_cast<const double*>(Qbuf.ptr);
        for (int i=0;i<nq;++i) for (int j=0;j<d;++j) Xq(i,j) = Xp[i*d + j];

        MatrixXd Kxs, Kss;
        kernel->K_cross(Xq, X, Kxs);   // nq x n
        kernel->K(Xq, Kss);            // nq x nq (only diag used)

        // mean = Kxs K^{-1} y = Kxs alpha
        VectorXd mean = Kxs * alpha;

        // v = solve(L, Kxs^T)  -> n x nq
        MatrixXd v = chol.matrixL().solve(Kxs.transpose());
        // var = diag(Kss) - sum(v^2, axis=0)
        VectorXd var(nq);
        for (int i=0;i<nq;++i) {
            double diagK = Kss(i,i);
            double sumv2 = v.col(i).squaredNorm();
            double base = std::max(1e-15, diagK - sumv2);
            if (!noiseless) base += std::exp(log_noise);
            var[i] = base;
        }

        // Return NumPy arrays
        py::array_t<double> mean_np(nq), var_np(nq);
        auto m = mean_np.mutable_unchecked<1>();
        auto s = var_np.mutable_unchecked<1>();
        for (int i=0;i<nq;++i) { m(i) = mean[i]; s(i) = var[i]; }
        return {mean_np, var_np};
    }

    // Simple marginal sampler: f(x) ~ N(mu(x), var(x)) i.i.d. across x (ok for our use)
    py::array_t<double> sample_marginals(
        py::array_t<double, py::array::c_style | py::array::forcecast> Xq_np,
        bool noiseless=true, uint64_t seed=0) {

        auto pr = predict(Xq_np, noiseless);
        py::array_t<double> mean_np = pr.first;
        py::array_t<double> var_np  = pr.second;
        auto m = mean_np.unchecked<1>();
        auto v = var_np.unchecked<1>();

        std::mt19937_64 rng(seed ? seed : std::random_device{}());
        std::normal_distribution<double> N(0.0, 1.0);
        const ssize_t n = m.shape(0);
        py::array_t<double> out(n);
        auto o = out.mutable_unchecked<1>();
        for (ssize_t i=0;i<n;++i) {
            double z = N(rng);
            o(i) = m(i) + std::sqrt(std::max(0.0, v(i))) * z;
        }
        return out;
    }

    py::dict get_hypers() const {
        std::vector<double> kp; kernel->get_params(kp);
        py::dict d;
        d["kernel_params"] = kp;
        d["log_noise"] = log_noise;
        return d;
    }
};

PYBIND11_MODULE(fastcarbs_core, m) {
    py::class_<GPRegression, py::smart_holder>(m, "GPRegression")
        .def(py::init<std::unique_ptr<Kernel>, double, double>(),
             py::arg("kernel"), py::arg("noise_var")=1e-2, py::arg("jitter")=1e-4)
        .def("set_data", &GPRegression::set_data)
        .def("fit", &GPRegression::fit, py::arg("num_steps")=1000, py::arg("lr")=1e-4)
        .def("predict", &GPRegression::predict, py::arg("Xq"), py::arg("noiseless")=true)
        .def("sample_marginals", &GPRegression::sample_marginals,
             py::arg("Xq"), py::arg("noiseless")=true, py::arg("seed")=0)
        .def("get_hypers", &GPRegression::get_hypers);

    py::class_<Kernel, py::smart_holder>(m, "Kernel");

    py::class_<Matern32ARD, Kernel, py::smart_holder>(m, "Matern32ARD")
        .def(py::init<size_t,double,double>(),
             py::arg("input_dim"), py::arg("init_lengthscale"),
             py::arg("init_variance")=1.0);

    py::class_<LinearKernel, Kernel, py::smart_holder>(m, "LinearKernel")
        .def(py::init<size_t,double>(),
             py::arg("input_dim"), py::arg("init_variance")=1.0);

    py::class_<SumKernel, Kernel, py::smart_holder>(m, "SumKernel")
        .def(py::init<std::unique_ptr<Kernel>, std::unique_ptr<Kernel>>(),
             py::arg("k1"), py::arg("k2"));

    py::class_<RBF1D, Kernel, py::smart_holder>(m, "RBF1D")
        .def(py::init<double,double>(),
             py::arg("init_lengthscale")=1.0, py::arg("init_variance")=1.0);
}
