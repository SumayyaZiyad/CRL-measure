# Implementation of the consistent record linkage (CRL) measure, and functions
# for its evaluation on a small example.
#
# Run from command line to execute the example.
#
# Peter Christen, November 2024
# -----------------------------------------------------------------------------

VERBOSE = False # True  # For more output

def precision(tp, fn, fp):
  if (tp + fp) > 0:
    return float(tp) / (tp + fp)
  else:
    return 0.0

def recall(tp, fn, fp):
  if (tp + fn) > 0:
    return float(tp) / (tp + fn)
  else:
    return 0.0

def f_measure(tp, fn, fp):
  if (tp + fn + fp) > 0:
    return 2.0*float(tp) / (2.0*tp + fp + fn)
  else:
    return 0.0

def f_star(tp, fn, fp):
  if (tp + fn + fp) > 0:
    return float(tp) / (tp + fp + fn)
  else:
    return 0.0

def false_negative_rate(tp, fn, fp):
  return 1 - recall(tp, fn, fp)

def false_discovery_rate(tp, fn, fp):
  return 1 - precision(tp, fn, fp)

# -----------------------------------------------------------------------------

def crl_measure(eps, S_list, m_class_label='m', n_class_label='n'):
  """Calculate the CRL-measure based on the given list of classification
     scores and class memberships (assumed to be as per defaults or set as
     function parameters).

     Returns the calculated CRL-measure value and the lower and upper
     classification thresholds between which the measure was calculated.
  """
  # Get the number of actual matches and actual non-matches in S_list
  #
  num_m = 0
  num_n = 0

  # First convert the list into a dictionary (associative array) where for
  # each unique classification score we obtain the counts of matches and
  # non-matches
  #
  S_dict = {}

  for (s_ij, c_ij) in S_list:
    if (s_ij not in S_dict):
      s_count_pair = [0, 0]
    else:
      s_count_pair = S_dict[s_ij]
    if (c_ij == 'm'):
      s_count_pair[0] += 1
      num_m += 1
    else:
      s_count_pair[1] += 1
      num_n += 1

    S_dict[s_ij] = s_count_pair

  assert num_m + num_n == len(S_list)

  # Sort the unique scores in reverse order (highest first)
  #
  unique_score_list = sorted(S_dict.keys(), reverse=True)
  num_unique_scores = len(unique_score_list)

  use_unique_s_ij_set = set()  # Unique scores used to calculate CRL-measure

  tp = 0  # All record pairs are initially classified as negatives
  fp = 0
  fn = num_m

  fnr = 1.0  # Initial resulting false negative rate

  k = 0  # Index into the score list

  F_list = []

  if (VERBOSE == True): print('-----------------------------')
  if (VERBOSE == True): print('  eps = %.3f' % (eps))
  if (VERBOSE == True): print('  Output: (k s_ij (tp, fn, fp) fnr fdr)')

  # First loop until fnr <= epsilon
  #
  while (fnr > eps) and (k < num_unique_scores):

    s_ij = unique_score_list[k]  # Next unique classification score
    score_m, score_n = S_dict[s_ij]  # Get counts of class memberships at s_ij
    k += 1

    tp += score_m
    fn -= score_m
    fp += score_n

    fnr = false_negative_rate(tp, fn, fp)

    if (VERBOSE == True): fdr = false_discovery_rate(tp, fn, fp)
    if (VERBOSE == True): print('    Loop 1:', k, s_ij, (tp, fn, fp), fnr, fdr)

  t_upper = s_ij  # Current score with a fnr <= epsilon
  t_lower = s_ij  # Ensure variable is set (in case loop 2 is never executed)

  if (VERBOSE == True): print('  t_upper:', t_upper)

  fdr = false_discovery_rate(tp, fn, fp)

  while (fdr <= eps) and (k < num_unique_scores):

    # Calculate the F-star measure for current confusion matrix (for which
    # both fnr <= epsilon and fdr <= epsilon)
    #
    F_list.append(f_star(tp, fn, fp))
    use_unique_s_ij_set.add(s_ij)

    s_ij = unique_score_list[k]  # Next unique classification score
    score_m, score_n = S_dict[s_ij]  # Get counts of class memberships at s_ij
    k += 1

    t_lower = s_ij  # Set to next value so we have non-zero interval

    tp += score_m
    fn -= score_m
    fp += score_n

    fdr = false_discovery_rate(tp, fn, fp)

    if (VERBOSE == True): fnr = false_negative_rate(tp, fn, fp)
    if (VERBOSE == True): print('    Loop 2:', k, s_ij, (tp, fn, fp), fdr, fnr)

  perc_scores_used = 100.0*k / len(unique_score_list)
    
  if (VERBOSE == True): print('  t_lower:', t_lower)
  if (VERBOSE == True): print('    %d unique scores in S_list, with %d used' \
                              % (len(unique_score_list),
                                 len(use_unique_s_ij_set)))
  if (VERBOSE == True): print('    Looped over %d from %d elements in S_dict' \
                              % (k, len(unique_score_list)) + \
                              ' (%.1f%%)' % (perc_scores_used))
  if (VERBOSE == True): print('-----------------------------')
  
  if (len(F_list) == 0):
    return 0.0, t_lower, t_upper
  else:
    # crl = sum(F_list) / len(F_list)  # Old version
    crl = (sum(F_list) / len(F_list)) * (t_upper - t_lower)
#    print('*** version 2 with threshold range: %.3f' % \
#          (sum(F_list) / len(F_list) * (t_upper - t_lower)))
    return crl, t_lower, t_upper

# =============================================================================
# If run from the command line, provide a simple example based on a small list
# of classification scores and class memberships for illustrative purposes
#
if (__name__ == '__main__'):

  S_list = [(0.99, 'm'), (0.98, 'm'), (0.96, 'm'), (0.95, 'm'), (0.95, 'm'),
            (0.94, 'm'), (0.93, 'm'), (0.91, 'm'), (0.90, 'm'), (0.88, 'm'),
            (0.85, 'm'), (0.84, 'm'), (0.82, 'm'), (0.81, 'n'), (0.80, 'm'),
            (0.78, 'm'), (0.77, 'm'), (0.76, 'm'), (0.75, 'm'), (0.74, 'm'),
            (0.73, 'n'), (0.72, 'n'), (0.72, 'n'), (0.71, 'n'), (0.70, 'm'),
            (0.69, 'n'), (0.68, 'n'), (0.66, 'n'), (0.65, 'n'), (0.65, 'n'),
            (0.64, 'n'), (0.63, 'n'), (0.62, 'n'), (0.61, 'n'), (0.60, 'n'),
            (0.59, 'n'), (0.58, 'n'), (0.57, 'n'), (0.56, 'n'), (0.55, 'n'),
            (0.55, 'n'), (0.54, 'n'), (0.54, 'n'), (0.54, 'n'), (0.53, 'n'),
            (0.53, 'n'), (0.52, 'n'), (0.52, 'n'), (0.51, 'n'), (0.50, 'n')]

  print()
  print('Example list S contains %d classification scores and class ' % \
        (len(S_list)) + 'memberships')

  num_m = 0
  num_n = 0

  for (s_ij, c_ij) in S_list:
    if (c_ij == 'm'):
      num_m += 1
    elif (c_ij == 'n'):
      num_n += 1
  assert num_m + num_n == len(S_list)

  print('  Of these, %d are actual matches and %d actual non-matches' % \
        (num_m, num_n))
  print()

  # Calculate confusion matrix for different classification thresholds
  #
  for t in [0.7, 0.8, 0.9]:
    tp = 0
    fn = 0
    fp = 0
    tn = 0

    for (s_ij, c_ij) in S_list:
      if (s_ij >= t) and (c_ij == 'm'):
        tp += 1
      elif (s_ij >= t) and (c_ij == 'n'):
        fp += 1
      elif (s_ij < t) and (c_ij == 'm'):
        fn += 1
      else:
        tn += 1

    assert tp+fp+fn+tn == len(S_list)

    print('For classification threshold t=%.2f we have the confusion matrix:' \
          % (t))
    print('  tp=%d, fn=%d, fp=%d, tn=%d' % (tp, fn, fp, tn))
    print()
    print('  Resulting performance measures:')
    print('  - Precision: %.3f' % (precision(tp, fn, fp)))
    print('  - Recall:    %.3f' % (recall(tp, fn, fp)))
    print('  - F-measure: %.3f' % (f_measure(tp, fn, fp)))
    print('  - F-star:    %.3f' % (f_star(tp, fn, fp)))
    print()

  # The CRL-measure is independent of the classification threshold
  #
  print('  - CRL-0.20 measure: %.3f (t_l=%.3f, t_u=%.3f)' % \
        crl_measure(0.2, S_list))
  print('  - CRL-0.10 measure: %.3f (t_l=%.3f, t_u=%.3f)' % \
        crl_measure(0.1, S_list))
  print('  - CRL-0.05 measure: %.3f (t_l=%.3f, t_u=%.3f)' % \
        crl_measure(0.05, S_list))
  print()

  S_list = [(0.4999, 'n'), (0.4999, 'n'), (0.5, 'm'), (0.5, 'm')]
  a,b,c =crl_measure(0.2, S_list)
  print(0.2, a, b, c, a/(c-b))
  a,b,c =crl_measure(0.1, S_list)
  print(-.1, a, b, c, a/(c-b))
  a,b,c =crl_measure(0.01, S_list)
  print(0.01, a, b, c, a/(c-b))

# End.
