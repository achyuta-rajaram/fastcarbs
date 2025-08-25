// fastcarbs_core.cpp

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

// ------------------------------- Utilities -------------------------------

inline Eigen::MatrixXd as_eigen_2d(py::array_t<double, py::array::c_style | py::array::forcecast> a) {
    if (a.ndim() != 2) throw std::invalid_argument("Expected a 2D array.");
    auto buf = a.request();
    auto* ptr = static_cast<double*>(buf.ptr);
    // Row-major map then copy into Eigen's default column-major MatrixXd
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> M(ptr, buf.shape[0], buf.shape[1]);
    return Eigen::MatrixXd(M);
}

inline Eigen::VectorXd as_eigen_vec(py::array_t<double, py::array::c_style | py::array::forcecast> a) {
    if (a.ndim() == 1) {
        auto buf = a.request();
        auto* ptr = static_cast<double*>(buf.ptr);
        Eigen::Map<const Eigen::VectorXd> v(ptr, buf.shape[0]);
        return Eigen::VectorXd(v);
    } else if (a.ndim() == 2) {
        auto buf = a.request();
        auto* ptr = static_cast<double*>(buf.ptr);
        if (buf.shape[1] == 1) {
            // (N,1) -> VectorXd
            Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1>> v(ptr, buf.shape[0]);
            return Eigen::VectorXd(v);
        } else if (buf.shape[0] == 1) {
            // (1,N) -> RowVectorXd, then transpose
            Eigen::Map<const Eigen::RowVectorXd> v(ptr, buf.shape[1]);
            return Eigen::VectorXd(v.transpose());
        } else {
            throw std::invalid_argument("y must be shape (N,), (N,1), or (1,N).");
        }
    } else {
        throw std::invalid_argument("y must be 1D or 2D.");
    }
}

inline py::array_t<double> eigen_to_numpy(const Eigen::VectorXd& v) {
    py::array_t<double> out(v.size());
    auto buf = out.request();
    double* ptr = static_cast<double*>(buf.ptr);
    Eigen::Map<Eigen::VectorXd>(ptr, v.size()) = v;
    return out;
}

// ------------------------------- Kernel Base -------------------------------

class Kernel {
public:
    virtual ~Kernel() = default;
    virtual std::string name() const = 0;
    virtual size_t input_dim() const = 0;

    // K(X, Z) : N x M
    virtual Eigen::MatrixXd K(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Z) const = 0;

    // K(X, X) : N x N
    virtual Eigen::MatrixXd K(const Eigen::MatrixXd& X) const {
        return K(X, X);
    }

    // Diagonal of K(X, X)
    virtual Eigen::VectorXd Kdiag(const Eigen::MatrixXd& X) const {
        Eigen::MatrixXd Kxx = K(X, X);
        return Kxx.diagonal();
    }

    // Parameterization: all positive hypers are internally stored as logs.
    virtual size_t num_params() const = 0;
    virtual void get_unconstrained(Eigen::VectorXd& u, size_t& off) const = 0;
    virtual void set_unconstrained(const Eigen::VectorXd& u, size_t& off) = 0;
    virtual void add_grad_wrt_unconstrained(const Eigen::MatrixXd& X,
                                            const Eigen::MatrixXd& B,
                                            Eigen::VectorXd& grad,
                                            size_t& off) const = 0;

    virtual py::dict hypers_dict() const = 0;
};

// ------------------------------- Matern32 ARD -------------------------------
// k(x,z) = sigma2 * (1 + sqrt(3) r) * exp(-sqrt(3) r)
// where r = sqrt( sum_j ((x_j - z_j)^2 / ell_j^2) )
class Matern32ARD final : public Kernel {
public:
    Matern32ARD(size_t input_dim, double init_lengthscale, double init_variance = 1.0)
        : D_(input_dim), log_ell_(Eigen::VectorXd::Constant(input_dim, std::log(init_lengthscale))),
          log_sigma2_(std::log(init_variance)) {}

    std::string name() const override { return "Matern32ARD"; }
    size_t input_dim() const override { return D_; }

    Eigen::MatrixXd K(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Z) const override {
        if (static_cast<size_t>(X.cols()) != D_ || static_cast<size_t>(Z.cols()) != D_) {
            throw std::invalid_argument("Matern32ARD: Input dimension mismatch.");
        }
        const double sigma2 = std::exp(log_sigma2_);
        const double a = std::sqrt(3.0);

        // Compute q = sum_j ( (x_j - z_j)^2 / ell_j^2 )
        Eigen::MatrixXd q = Eigen::MatrixXd::Zero(X.rows(), Z.rows());
        for (size_t j = 0; j < D_; ++j) {
            const double inv_l2 = std::exp(-2.0 * log_ell_(j)); // 1 / ell^2
            Eigen::VectorXd xj = X.col(static_cast<int>(j));
            Eigen::VectorXd zj = Z.col(static_cast<int>(j));
            Eigen::VectorXd xj2 = xj.array().square().matrix();
            Eigen::VectorXd zj2 = zj.array().square().matrix();
            Eigen::MatrixXd S = xj2.replicate(1, Z.rows())
                              + zj2.transpose().replicate(X.rows(), 1)
                              - 2.0 * (xj * zj.transpose());
            q.noalias() += inv_l2 * S;
        }
        Eigen::MatrixXd r = q.array().max(0.0).sqrt().matrix();
        Eigen::ArrayXXd E = (-a * r.array()).exp();       // exp(-a r)
        Eigen::ArrayXXd base = (1.0 + a * r.array()) * E; // (1 + a r) exp(-a r)
        Eigen::MatrixXd K = (sigma2 * base).matrix();
        return K;
    }

    size_t num_params() const override { return D_ + 1; }

    void get_unconstrained(Eigen::VectorXd& u, size_t& off) const override {
        for (size_t j = 0; j < D_; ++j) u(off++) = log_ell_(static_cast<int>(j));
        u(off++) = log_sigma2_;
    }

    void set_unconstrained(const Eigen::VectorXd& u, size_t& off) override {
        for (size_t j = 0; j < D_; ++j) log_ell_(static_cast<int>(j)) = u(off++);
        log_sigma2_ = u(off++);
    }

    void add_grad_wrt_unconstrained(const Eigen::MatrixXd& X,
                                    const Eigen::MatrixXd& B,
                                    Eigen::VectorXd& grad,
                                    size_t& off) const override {
        // Compute helpers on X
        const int N = static_cast<int>(X.rows());
        const double a = std::sqrt(3.0);
        const double sigma2 = std::exp(log_sigma2_);

        // Precompute per-dimension squared-distance matrices S_j and q = sum S_j / ell^2
        std::vector<Eigen::MatrixXd> Sj(D_);
        Eigen::MatrixXd q = Eigen::MatrixXd::Zero(N, N);
        for (size_t j = 0; j < D_; ++j) {
            Eigen::VectorXd xj = X.col(static_cast<int>(j));
            Eigen::VectorXd xj2 = xj.array().square().matrix();
            Eigen::MatrixXd S = xj2.replicate(1, N) + xj2.transpose().replicate(N, 1) - 2.0 * (xj * xj.transpose());
            Sj[j] = std::move(S);
        }
        for (size_t j = 0; j < D_; ++j) {
            const double inv_l2 = std::exp(-2.0 * log_ell_(static_cast<int>(j)));
            q.noalias() += inv_l2 * Sj[j];
        }
        Eigen::MatrixXd r = q.array().max(0.0).sqrt().matrix();
        Eigen::ArrayXXd E = (-a * r.array()).exp();              // exp(-a r)
        Eigen::ArrayXXd base = (1.0 + a * r.array()) * E;        // (1 + a r) exp(-a r)

        // dK / d sigma2 = base
        Eigen::MatrixXd dK_dsigma2 = base.matrix();
        double g_sigma2 = 0.5 * (B.cwiseProduct(dK_dsigma2)).sum(); // dL/dsigma2
        // chain to log_sigma2
        grad(off + static_cast<int>(D_)) += g_sigma2 * std::exp(log_sigma2_);

        // dK / d ell_j = sigma2 * a^2 * exp(-a r) * S_j / ell_j^3
        const double a2 = a * a;
        for (size_t j = 0; j < D_; ++j) {
            const double ell = std::exp(log_ell_(static_cast<int>(j)));
            const double inv_l3 = 1.0 / (ell * ell * ell);
            Eigen::MatrixXd dK_dell = (sigma2 * a2) * (E.matrix().cwiseProduct(Sj[j])) * inv_l3;
            double g_ell = 0.5 * (B.cwiseProduct(dK_dell)).sum(); // dL/dell
            grad(off + static_cast<int>(j)) += g_ell * ell;       // chain to log_ell
        }

        off += D_ + 1;
    }

    py::dict hypers_dict() const override {
        py::dict d;
        d["kernel"] = py::str("Matern32ARD");
        py::array_t<double> ls(D_);
        {
            auto buf = ls.request();
            double* ptr = static_cast<double*>(buf.ptr);
            for (size_t j = 0; j < D_; ++j) ptr[j] = std::exp(log_ell_(static_cast<int>(j)));
        }
        d["lengthscales"] = ls;
        d["variance"] = std::exp(log_sigma2_);
        return d;
    }

private:
    size_t D_;
    Eigen::VectorXd log_ell_;
    double log_sigma2_;
};

// ------------------------------- Linear Kernel -------------------------------
// k(x,z) = sigma2 * x . z
class LinearKernel final : public Kernel {
public:
    LinearKernel(size_t input_dim, double init_variance = 1.0)
        : D_(input_dim), log_sigma2_(std::log(init_variance)) {}

    std::string name() const override { return "Linear"; }
    size_t input_dim() const override { return D_; }

    Eigen::MatrixXd K(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Z) const override {
        if (static_cast<size_t>(X.cols()) != D_ || static_cast<size_t>(Z.cols()) != D_) {
            throw std::invalid_argument("LinearKernel: Input dimension mismatch.");
        }
        const double sigma2 = std::exp(log_sigma2_);
        Eigen::MatrixXd K = sigma2 * (X * Z.transpose());
        return K;
    }

    size_t num_params() const override { return 1; }
    void get_unconstrained(Eigen::VectorXd& u, size_t& off) const override { u(off++) = log_sigma2_; }
    void set_unconstrained(const Eigen::VectorXd& u, size_t& off) override { log_sigma2_ = u(off++); }

    void add_grad_wrt_unconstrained(const Eigen::MatrixXd& X,
                                    const Eigen::MatrixXd& B,
                                    Eigen::VectorXd& grad,
                                    size_t& off) const override {
        // dK/dsigma2 = X X^T
        Eigen::MatrixXd G = X * X.transpose();
        double g_sigma2 = 0.5 * (B.cwiseProduct(G)).sum();
        grad(off) += g_sigma2 * std::exp(log_sigma2_); // chain to log_sigma2
        off += 1;
    }

    py::dict hypers_dict() const override {
        py::dict d;
        d["kernel"] = py::str("Linear");
        d["variance"] = std::exp(log_sigma2_);
        return d;
    }

private:
    size_t D_;
    double log_sigma2_;
};

// ------------------------------- Sum Kernel -------------------------------

class SumKernel final : public Kernel {
public:
    SumKernel(std::unique_ptr<Kernel> k1, std::unique_ptr<Kernel> k2)
        : k1_(std::move(k1)), k2_(std::move(k2)) {
        if (k1_->input_dim() != k2_->input_dim())
            throw std::invalid_argument("SumKernel: subkernels must have same input_dim.");
    }

    std::string name() const override { return "Sum"; }
    size_t input_dim() const override { return k1_->input_dim(); }

    Eigen::MatrixXd K(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Z) const override {
        return k1_->K(X, Z) + k2_->K(X, Z);
    }

    size_t num_params() const override { return k1_->num_params() + k2_->num_params(); }

    void get_unconstrained(Eigen::VectorXd& u, size_t& off) const override {
        k1_->get_unconstrained(u, off);
        k2_->get_unconstrained(u, off);
    }

    void set_unconstrained(const Eigen::VectorXd& u, size_t& off) override {
        k1_->set_unconstrained(u, off);
        k2_->set_unconstrained(u, off);
    }

    void add_grad_wrt_unconstrained(const Eigen::MatrixXd& X,
                                    const Eigen::MatrixXd& B,
                                    Eigen::VectorXd& grad,
                                    size_t& off) const override {
        k1_->add_grad_wrt_unconstrained(X, B, grad, off);
        k2_->add_grad_wrt_unconstrained(X, B, grad, off);
    }

    py::dict hypers_dict() const override {
        py::dict d;
        d["kernel"] = py::str("Sum");
        d["k1"] = k1_->hypers_dict();
        d["k2"] = k2_->hypers_dict();
        return d;
    }

private:
    std::unique_ptr<Kernel> k1_, k2_;
};

// ------------------------------- RBF 1D -------------------------------
// k(x,z) = sigma2 * exp( -0.5 * (x - z)^2 / ell^2 ) ; 1D only
class RBF1D final : public Kernel {
public:
    RBF1D(double init_lengthscale = 1.0, double init_variance = 1.0)
        : log_ell_(std::log(init_lengthscale)), log_sigma2_(std::log(init_variance)) {}

    std::string name() const override { return "RBF1D"; }
    size_t input_dim() const override { return 1; }

    Eigen::MatrixXd K(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Z) const override {
        if (X.cols() != 1 || Z.cols() != 1)
            throw std::invalid_argument("RBF1D expects inputs with shape (N,1).");
        const double ell = std::exp(log_ell_);
        const double sigma2 = std::exp(log_sigma2_);
        const double inv_ell2 = 1.0 / (ell * ell);

        Eigen::VectorXd x = X.col(0);
        Eigen::VectorXd z = Z.col(0);
        Eigen::VectorXd x2 = x.array().square().matrix();
        Eigen::VectorXd z2 = z.array().square().matrix();
        Eigen::MatrixXd S = x2.replicate(1, Z.rows()) + z2.transpose().replicate(X.rows(), 1) - 2.0 * (x * z.transpose());
        Eigen::MatrixXd K = (sigma2 * (-0.5 * inv_ell2 * S.array()).exp()).matrix();
        return K;
    }

    size_t num_params() const override { return 2; }
    void get_unconstrained(Eigen::VectorXd& u, size_t& off) const override { u(off++) = log_ell_; u(off++) = log_sigma2_; }
    void set_unconstrained(const Eigen::VectorXd& u, size_t& off) override { log_ell_ = u(off++); log_sigma2_ = u(off++); }

    void add_grad_wrt_unconstrained(const Eigen::MatrixXd& X,
                                    const Eigen::MatrixXd& B,
                                    Eigen::VectorXd& grad,
                                    size_t& off) const override {
        if (X.cols() != 1) throw std::invalid_argument("RBF1D expects inputs with shape (N,1).");
        const int N = static_cast<int>(X.rows());
        const double ell = std::exp(log_ell_);
        const double sigma2 = std::exp(log_sigma2_);
        const double inv_ell2 = 1.0 / (ell * ell);

        Eigen::VectorXd x = X.col(0);
        Eigen::VectorXd x2 = x.array().square().matrix();
        Eigen::MatrixXd S = x2.replicate(1, N) + x2.transpose().replicate(N, 1) - 2.0 * (x * x.transpose());
        Eigen::MatrixXd base = (-0.5 * inv_ell2 * S.array()).exp().matrix(); // exp(-0.5 S/ell^2)

        // dK/dsigma2 = base
        Eigen::MatrixXd dK_dsigma2 = base;
        double g_sigma2 = 0.5 * (B.cwiseProduct(dK_dsigma2)).sum();
        grad(off + 1) += g_sigma2 * sigma2; // chain to log_sigma2

        // dK/dell = sigma2 * base * (S / ell^3)
        double inv_ell3 = 1.0 / (ell * ell * ell);
        Eigen::MatrixXd dK_dell = sigma2 * base.cwiseProduct(S) * inv_ell3;
        double g_ell = 0.5 * (B.cwiseProduct(dK_dell)).sum();
        grad(off + 0) += g_ell * ell;       // chain to log_ell

        off += 2;
    }

    py::dict hypers_dict() const override {
        py::dict d;
        d["kernel"] = py::str("RBF1D");
        d["lengthscale"] = std::exp(log_ell_);
        d["variance"] = std::exp(log_sigma2_);
        return d;
    }

private:
    double log_ell_;
    double log_sigma2_;
};

// ------------------------------- GP Regression -------------------------------

class GPRegression {
public:
    GPRegression(std::unique_ptr<Kernel> kernel, double noise_var = 1e-2, double jitter = 1e-4)
        : kernel_(std::move(kernel)), log_noise_(std::log(noise_var)), jitter_(jitter)
    {
        if (!kernel_) throw std::invalid_argument("Kernel must not be null.");
    }

    void set_data(py::array_t<double> X_in, py::array_t<double> y_in) {
        X_ = as_eigen_2d(X_in);
        y_ = as_eigen_vec(y_in);
        const auto N = X_.rows();
        if (y_.rows() != N)
            throw std::invalid_argument("set_data: X and y have incompatible first dimension.");
        // invalidate cache
        L_.resize(0,0);
    }

    // Adam on MAP objective: NLML + (-log prior(noise))
    void fit(int num_steps = 1000, double lr = 1e-4) {
        if (X_.size() == 0) throw std::runtime_error("No data set. Call set_data first.");
        const int N = static_cast<int>(X_.rows());

        // flatten params: [kernel params (unconstrained), log_noise]
        const size_t Pk = kernel_->num_params();
        const size_t P = Pk + 1;
        Eigen::VectorXd u(P);
        {
            size_t off = 0;
            kernel_->get_unconstrained(u, off);
            u(off++) = log_noise_;
        }

        // Adam state
        const double beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
        Eigen::VectorXd m = Eigen::VectorXd::Zero(P);
        Eigen::VectorXd v = Eigen::VectorXd::Zero(P);

        // constants for LogNormal prior on noise (PyroSample in your code)
        const double mu = std::log(1e-2);
        const double sigma = 0.5;
        const double sigma2 = sigma * sigma;
        const double halfNlog2pi = 0.5 * N * std::log(2.0 * M_PI);

        for (int t = 1; t <= num_steps; ++t) {
            // unpack
            size_t off = 0;
            kernel_->set_unconstrained(u, off);
            log_noise_ = u(off++);
            const double noise = std::exp(log_noise_);

            // K = K(X,X) + (noise + jitter) I
            Eigen::MatrixXd K = kernel_->K(X_);
            K.diagonal().array() += (noise + jitter_);

            // Cholesky
            Eigen::LLT<Eigen::MatrixXd> llt(K);
            if (llt.info() != Eigen::Success) {
                throw std::runtime_error("Cholesky factorization failed. Try larger jitter.");
            }
            Eigen::MatrixXd L = llt.matrixL();

            // alpha = K^{-1} y via solves
            Eigen::VectorXd alpha = llt.solve(y_);

            // logdet K = 2 * sum(log diag(L))
            double logdet = 0.0;
            for (int i = 0; i < L.rows(); ++i) logdet += std::log(L(i,i));
            logdet *= 2.0;

            // NLML
            double nll = 0.5 * y_.dot(alpha) + 0.5 * logdet + halfNlog2pi;

            // Negative log prior for noise ~ LogNormal(mu, sigma)
            // nlp = log(noise) + log(sigma*sqrt(2pi)) + (log(noise)-mu)^2/(2*sigma^2)
            const double logn = log_noise_;
            double nlp_noise = logn + std::log(sigma * std::sqrt(2.0*M_PI)) + (logn - mu)*(logn - mu) / (2.0 * sigma2);

            double loss = nll + nlp_noise;

            // -------- gradient ----------
            // B = alpha alpha^T - K^{-1}
            Eigen::MatrixXd Kinv = llt.solve(Eigen::MatrixXd::Identity(N, N));
            Eigen::MatrixXd B = alpha * alpha.transpose() - Kinv;

            Eigen::VectorXd g = Eigen::VectorXd::Zero(P);
            // kernel hypers grad (w.r.t log params)
            off = 0;
            kernel_->add_grad_wrt_unconstrained(X_, B, g, off);

            // noise grad (marginal likelihood term): 0.5 * tr(B * dK/dnoise) * dnoise/dlognoise
            // dK/dnoise = I
            double dL_dnoise = 0.5 * B.trace();
            g(off) += dL_dnoise * noise;

            // add prior grad wrt log noise: d/d(log n) [log n + (log n - mu)^2/(2 sigma^2)] = 1 + (logn - mu)/sigma^2
            g(off) += 1.0 + (logn - mu) / sigma2;
            // (constant term log(sigma sqrt(2pi)) drops)

            // Adam update
            m = beta1 * m + (1.0 - beta1) * g;
            v = beta2 * v + (1.0 - beta2) * g.array().square().matrix();

            Eigen::VectorXd mhat = m / (1.0 - std::pow(beta1, t));
            Eigen::VectorXd vhat = v / (1.0 - std::pow(beta2, t));

            Eigen::ArrayXd step = lr * mhat.array() / (vhat.array().sqrt() + eps);
            u -= step.matrix();
            
            // (optional) could store last L for warm-start predict; we recompute in predict anyway.
        }

        // write back final params
        size_t off2 = 0;
        kernel_->set_unconstrained(u, off2);
        log_noise_ = u(off2++);

        // cache last K chol for predict-only scenario? (not strictly necessary)
        L_.resize(0,0);
    }

    // predict: returns (mean, var) at Xq, marginal (diagonal) only, noiseless or noisy.
    std::pair<py::array_t<double>, py::array_t<double>>
    predict(py::array_t<double> Xq_in, bool noiseless = true) const {
        if (X_.size() == 0) throw std::runtime_error("No data set. Call set_data first.");

        Eigen::MatrixXd Xq = as_eigen_2d(Xq_in);
        if (Xq.cols() != X_.cols())
            throw std::invalid_argument("predict: Xq has wrong feature dimension.");

        const double noise = std::exp(log_noise_);
        // Build training K and chol
        Eigen::MatrixXd K = kernel_->K(X_);
        K.diagonal().array() += (noise + jitter_);
        Eigen::LLT<Eigen::MatrixXd> llt(K);
        if (llt.info() != Eigen::Success)
            throw std::runtime_error("Cholesky factorization failed in predict().");

        Eigen::MatrixXd L = llt.matrixL();

        // alpha = K^{-1} y
        Eigen::VectorXd alpha = llt.solve(y_);

        // cross-cov and self-cov
        Eigen::MatrixXd KfXq = kernel_->K(X_, Xq);        // N x M
        Eigen::MatrixXd Kqq  = kernel_->K(Xq, Xq);        // M x M

        // mean = KfXq^T * alpha
        Eigen::VectorXd mean = (KfXq.transpose() * alpha);

        // v = solve(L, KfXq)
        Eigen::MatrixXd v = L.triangularView<Eigen::Lower>().solve(KfXq);
        // var_diag = diag(Kqq - v^T v)
        Eigen::VectorXd var = Kqq.diagonal() - (v.array().square().colwise().sum()).matrix().transpose();

        if (!noiseless) var.array() += noise;

        return {eigen_to_numpy(mean), eigen_to_numpy(var)};
    }

    // sample from marginal normals N(mean_i, var_i)
    py::array_t<double> sample_marginals(py::array_t<double> Xq_in, bool noiseless = true, int seed = 0) const {
        auto mv = predict(Xq_in, noiseless);
        Eigen::VectorXd mean = as_eigen_vec(mv.first);
        Eigen::VectorXd var  = as_eigen_vec(mv.second);

        std::mt19937_64 rng(static_cast<uint64_t>(seed));
        std::normal_distribution<double> N01(0.0, 1.0);

        Eigen::VectorXd s(mean.size());
        for (int i = 0; i < mean.size(); ++i) {
            double stdv = std::sqrt(std::max(0.0, var(i)));
            s(i) = mean(i) + stdv * N01(rng);
        }
        return eigen_to_numpy(s);
    }

    py::dict get_hypers() const {
        py::dict d;
        d["noise"]  = std::exp(log_noise_);
        d["jitter"] = jitter_;
        d["kernel"] = kernel_->hypers_dict();
        return d;
    }

private:
    std::unique_ptr<Kernel> kernel_;
    Eigen::MatrixXd X_;
    Eigen::VectorXd y_;
    mutable Eigen::MatrixXd L_; // not used now (placeholder for caching)
    double log_noise_;
    double jitter_;
};

// ------------------------------- PYBIND11 module -------------------------------

PYBIND11_MODULE(fastcarbs_core, m) {
    py::class_<GPRegression, py::smart_holder>(m, "GPRegression")
        .def(py::init<std::unique_ptr<Kernel>, float, float>(),
             py::arg("kernel"), py::arg("noise_var")=1e-2, py::arg("jitter")=1e-4)
        .def("set_data", &GPRegression::set_data)
        .def("fit", &GPRegression::fit, py::arg("num_steps")=1000, py::arg("lr")=1e-4)
        .def("predict", &GPRegression::predict, py::arg("Xq"), py::arg("noiseless")=true)
        .def("sample_marginals", &GPRegression::sample_marginals,
             py::arg("Xq"), py::arg("noiseless")=true, py::arg("seed")=0)
        .def("get_hypers", &GPRegression::get_hypers);

    py::class_<Kernel, py::smart_holder>(m, "Kernel");

    py::class_<Matern32ARD, Kernel, py::smart_holder>(m, "Matern32ARD")
        .def(py::init<size_t,float,float>(),
             py::arg("input_dim"), py::arg("init_lengthscale"),
             py::arg("init_variance")=1.0);

    py::class_<LinearKernel, Kernel, py::smart_holder>(m, "LinearKernel")
        .def(py::init<size_t,float>(),
             py::arg("input_dim"), py::arg("init_variance")=1.0);

    py::class_<SumKernel, Kernel, py::smart_holder>(m, "SumKernel")
        .def(py::init<std::unique_ptr<Kernel>, std::unique_ptr<Kernel>>(),
             py::arg("k1"), py::arg("k2"));

    py::class_<RBF1D, Kernel, py::smart_holder>(m, "RBF1D")
        .def(py::init<float,float>(),
             py::arg("init_lengthscale")=1.0, py::arg("init_variance")=1.0);
}
