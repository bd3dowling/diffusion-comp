from particles.state_space_models import GuidedPF


class LikelihoodGuidedPF(GuidedPF):
    def logG(self, t, xp, x):
        if t == 0:
            return self.ssm.PY(t, xp, x).logpdf(self.data[t])

        # Define the log-weights based on the ratio of likelihoods
        # NOTE: The `t` argument to `PY` doesn't matter; `xp` key for `g_y_t1`
        g_y_t = self.ssm.PY(t, xp, x).logpdf(self.data[t])
        g_y_t1 = self.ssm.PY(t - 1, xp, xp).logpdf(self.data[t - 1])

        return g_y_t - g_y_t1
