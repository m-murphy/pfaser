// The input data is a vector 'y' of length 'N'.
data {
  int<lower=0> N; // total observations
  int<lower=0> L; // total Loci
  int<lower=0> COI; // complexity of infection
  vector[N] y; // read counts at each allele
  int alleles[L]; // array of number of alleles at each locus
  real beta_alpha_scale; // scaling factor for alpha parameter in dirichlet prior for beta vectors
  real theta_C; // sparsity scaling factor for alpha parameter in dirichlet prior for theta
  real pi_alpha; // alpha parameter in dirichlet prior for pi vectors
  real lam_alpha; // alpha parameter in beta prior for lambda
  real lam_beta; // beta parameter in beta prior for lambda
}

transformed data {
   vector[COI] theta_alpha;
   vector[N * COI] beta_alpha;
   int alpha_counter = 1;
   int idx_counter = 1;

   for (l in 1:L) {
     for (c in 1:COI) {
       for (a in 1:alleles[l]) {
         beta_alpha[alpha_counter] = 1.0 / (alleles[l] * beta_alpha_scale);
         alpha_counter += 1;
       }
     }
   }

   for (c in 1:COI) {
     // theta ordered from smallest to largest, so make beginning of prior sparse
     theta_alpha[c] = theta_C / (COI - c + 1);
   }
}

parameters {
  positive_ordered[COI] pre_theta;
  real<lower=0, upper=.05> lam;
  // can't represent simplices as ragged data structures, so use gamma distributed params
  // collection of normalized gammas -> dirichlet dist
  vector<lower=0>[N * COI] gamma_beta;  // beta -> strain identity simplex
  vector<lower=0>[N] gamma_pi; // pi -> false positive count distribution
}

transformed parameters {
  simplex[COI] theta = pre_theta / sum(pre_theta);
}

model {
  int beta_pos = 1;
  int pi_pos = 1;
  for (l in 1:L) {
    vector[alleles[l]] allele_freq = rep_vector(0.0, alleles[l]);
    vector[alleles[l]] fpi = segment(gamma_pi, pi_pos, alleles[l]);
    vector[alleles[l]] dat = segment(y, pi_pos, alleles[l]);
    for (c in 1:COI) {
      vector[alleles[l]] beta = segment(gamma_beta, beta_pos, alleles[l]);
      beta = beta / sum(beta);
      allele_freq += theta[c] * beta;
      beta_pos += alleles[l];
    }
    fpi = fpi / sum(fpi);
    allele_freq = (allele_freq * (1 - lam)) + (fpi * lam);
    target += sum(dat .* log(allele_freq));
    pi_pos += alleles[l];
  }

  target += beta_lpdf(lam | lam_alpha, lam_beta);
  target += gamma_lpdf(pre_theta | theta_alpha, 1);
  target += gamma_lpdf(gamma_beta | beta_alpha, 1);
  target += gamma_lpdf(gamma_pi | pi_alpha, 1);
}

generated quantities {
  matrix[L, COI] strain_identity;
  matrix[COI, N] strain_dist;
  {
    int beta_pos = 1;
    int locus_coi_pos = 1;
    for (l in 1:L) {
      for (c in 1:COI) {
        real max_el_val = 0;

        vector[alleles[l]] beta = segment(gamma_beta, beta_pos, alleles[l]);
        beta = beta / sum(beta);

        for (a in 1:alleles[l]) {
          if (beta[a] > max_el_val) {
            max_el_val = beta[a];
            strain_identity[l, c] = a;
          }
          strain_dist[c, locus_coi_pos + (a - 1)] = beta[a];
          beta_pos += 1;
        }
      }
      locus_coi_pos += alleles[l];
    }
  }
}
