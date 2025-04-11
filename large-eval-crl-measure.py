# Evaluate the clr-measure using several generated sets of record linkage
# results
#
# Run in two ways, either as:
#
#   python3 large-eval-crl-measure.py [num_m] [num_n]
# or:
#   python3 large-eval-crl-measure.py [res_file_name]
#
# where:
# - num_m  is the number of matches to be generated in all test data sets
# - num_n  is the number of non-matches to be generated in all test data sets
#  -res_file_name is a CSV or CSV.GZ file where in each row there is a
#   record identifier pair, a classification score value, and a class
#   membership value (1 for a match and 0 for a non-match).

# Peter Christen, November 2024 to April 2025
# -----------------------------------------------------------------------------

import gzip
import random
import sys

import numpy

import crl_measure

eps_list = []
for eps in range(1,100):
  eps_list.append(float(eps) / 100)
print(eps_list)

#eps_list = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, \
#            0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.0]

# -----------------------------------------------------------------------------
# The functions that will generate the data sets, each will return a list the
# contains classification scores and class labels ('m' and 'n')

def perfect_classification(num_m, num_n):
  """Generate num_m matches with a classification score of 1.0 and num_n
     non-matches with a classification score of 0.0. This corresponds to a
     perfect classification outcomes.
  """

  S_list = []

  for i in range(num_m):
    S_list.append((1.0, 'm'))

  for i in range(num_n):
    S_list.append((0.0, 'n'))

  return S_list

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def excellent_classification(num_m, num_n):
  """Generate num_m matches and num_n non-matches, where 95% of matches have
     a classification score of at least 0.9 and 5% a score of at least 0.7 but
     below 0.9. For non-matches, 5% have a score of at least 0.7 but below
     0.9, and 95% a score below 0.7
  """

  # Calculate numbers of matches and non-matches for each segment of 0.1
  # scores
  #
  num_m07 = int(0.05 * num_m)  # Number of scores >= 0.7 and < 0.9
  num_m09 = num_m - num_m07    # Number of scores >= 0.9

  num_n07 = int(0.05 * num_n)  # Number of scores >= 0.7 and < 0.9
  num_n00 = num_n - num_n07    # Number of scores < 0.7

  S_list = []

  # Loop over the different categories until the required numbers of matches
  # or non-matches are selected
  #
  for i in range(num_m09):
    s = float(random.randint(900000, 1000000)) / 1000000  # 0.9 to 1 inclusive
    assert (s >= 0.9) and (s <= 1.0), s 
    S_list.append((s, 'm'))
  for i in range(num_m07):
    s = float(random.randint(700000, 899999)) / 1000000  # 0.7 to 0.9
    assert (s >= 0.7) and (s < 0.9), s
    S_list.append((s, 'm'))

  for i in range(num_n07):
    s = float(random.randint(700000, 899999)) / 1000000  # 0.7 to 0.9
    assert (s >= 0.7) and (s < 0.9), s 
    S_list.append((s, 'n'))
  for i in range(num_n00):
    s = float(random.randint(0, 699999)) / 1000000  # 0.0 to 0.7
    assert (s < 0.7), s 
    S_list.append((s, 'n'))

  # Check the required numbers of matches and non-matches have been generated
  #
  check_m = 0
  check_n = 0
  for (s,c) in S_list:
    if c == 'm': check_m +=1
    else: check_n += 1
  assert num_m == check_m and num_n == check_n
  
  return S_list

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def good_classification(num_m, num_n):
  """Generate num_m matches and num_n non-matches, where 85% of matches have
     a classification score of at least 0.9, 10% a score of at least 0.8 but
     below 0.9, and 5% a score of at least 0.7 but below 0.8. For non-matches,
     5% have a score of at least 0.8 but below 0.9, 10% a score of at least
     0.7 but below 0.8, and 85% a score below 0.7
  """

  # Calculate numbers of matches and non-matches for each segment of 0.1
  # scores
  #
  num_m07 = int(0.05 * num_m)          # Number of scores >= 0.7 and < 0.8
  num_m08 = int(0.1 * num_m)           # Number of scores >= 0.8 and < 0.9
  num_m09 = num_m - num_m07 - num_m08  # Number of scores >= 0.9

  num_n08 = int(0.05 * num_n)         # Number of scores >= 0.8 and < 0.9
  num_n07 = int(0.1 * num_n)          # Number of scores >= 0.7 and < 0.8
  num_n00 = num_n - num_n08 - num_n07 # Number of scores < 0.7
  
  S_list = []

  # Loop over the different categories until the required numbers of matches
  # or non-matches are selected
  #
  for i in range(num_m09):
    s = float(random.randint(900000, 1000000)) / 1000000  # 0.9 to 1 inclusive
    assert (s >= 0.9) and (s <= 1.0), s 
    S_list.append((s, 'm'))
  for i in range(num_m08):
    s = float(random.randint(800000, 899999)) / 1000000  # 0.8 to 0.9
    assert (s >= 0.8) and (s < 0.9), s
    S_list.append((s, 'm'))
  for i in range(num_m07):
    s = float(random.randint(700000, 799999)) / 1000000  # 0.7 to 0.8
    assert (s >= 0.7) and (s < 0.8), s
    S_list.append((s, 'm'))

  for i in range(num_n08):
    s = float(random.randint(800000, 899999)) / 1000000  # 0.8 to 0.9
    assert (s >= 0.8) and (s < 0.9), s 
    S_list.append((s, 'n'))
  for i in range(num_n07):
    s = float(random.randint(700000, 799999)) / 1000000  # 0.7 to 0.8
    assert (s >= 0.7) and (s < 0.8), s 
    S_list.append((s, 'n'))
  for i in range(num_n00):
    s = float(random.randint(0, 699999)) / 1000000  # 0.0 to 0.7
    assert (s < 0.7), s 
    S_list.append((s, 'n'))

  # Check the required numbers of matches and non-matches have been generated
  #
  check_m = 0
  check_n = 0
  for (s,c) in S_list:
    if c == 'm': check_m +=1
    else: check_n += 1
  assert num_m == check_m and num_n == check_n
  
  return S_list

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def total_random(num_m, num_n):
  """Generate num_m matches and num_n non-matches where their classification
     scores are randomly selected in the range 0 to 1.
  """

  S_list = []

  for i in range(num_m):
    s = float(random.randint(0, 1000000)) / 1000000  # 0 to 1 inclusive
    assert (s >= 0.0) and (s <= 1.0)
    S_list.append((s, 'm'))

  for i in range(num_n):
    s = float(random.randint(0, 1000000)) / 1000000
    assert (s >= 0.0) and (s <= 1.0)
    S_list.append((s, 'n'))

  return S_list

# -----------------------------------------------------------------------------
# Function to get the confusion matrix at a given threshold

def get_threshold_conf_matrix(S_list, class_thres = 0.5):
  """Calculate the counts for the confusion matrix at the given threshold
     (with a default of 0.5) for the given list of classification scores and
     class labels.
     Returns the counts of tp, fn, fp, and tn.
  """

  tp, fn, fp, tn = 0, 0, 0, 0

  for (class_score, class_label) in S_list:
    assert class_label in ['m', 'n'], class_label
    if (class_score >= class_thres):
      if (class_label == 'm'):
        tp += 1
      else:
        fp += 1
    else:
      if (class_label == 'm'):
        fn += 1
      else:
        tn += 1

  assert len(S_list) == tp + fn + fp + tn

  return tp, fn, fp, tn

# -----------------------------------------------------------------------------
# Function to get the optimal result for the given measure, and return that
# result together with the best threshold where it was obtained

def get_best_result(S_list, perf_meas_funct):
  """Calculate the best possible (largest) value for the given performance
     measure and given list of classification scores and class labels.
     Return that value and the corresponding threshold where it was obtained.
  """

  best_perf =  -1.0  # Best obtained performance
  best_thres = -1.0  # Classification threshold of best obtained performance

  all_m = 0  # Get number of matches in S_list
  num_all = len(S_list)
  
  for (class_score, class_label) in S_list:
    assert class_label in ['m', 'n']
    if (class_label == 'm'):
      all_m += 1
  all_n = num_all - all_m

  # Assume all record pairs are initially classified as non-matches
  #
  tp = 0
  fn = all_m
  fp = 0
  tn = all_n

  S_sorted = sorted(S_list, reverse=True)  # Sort with largest first
  
  for (class_score, class_label) in S_sorted:
    curr_thres = class_score

    if (class_label == 'm'):
      tp += 1
      fn -= 1
    else:
      fp += 1
      tn -= 1

    assert tp+fp+fn+tn == num_all

    perf = perf_meas_funct(tp, fn, fp)
    if (perf > best_perf):
      best_perf =  perf
      best_thres = curr_thres

  return best_perf, best_thres

# -----------------------------------------------------------------------------
# Calculate the area under the precision recall curve (AUC-PR)

def calc_auc_pr(S_list, prec_funct, reca_funct):
  """Calculate the area under the precision recall curve (AUC-PR) for the given
     list of classification scores and class labels, and return the calculated
     value.
  """

  S_sorted = sorted(S_list, reverse=True)  # Sort with largest first

  all_m = 0  # Get number of matches in S_list
  num_all = len(S_list)
  
  for (class_score, class_label) in S_list:
    assert class_label in ['m', 'n']
    if (class_label == 'm'):
      all_m += 1
      
  # Calculate the AUC-PR (follow sklearn.metrics.average_precision_score)
  # In the prec_list and reca_list we have high precision and low recall
  # first (assuming we start with the highest classification scores), while
  # at the end we have high recall and low precision. To be able to calculate
  # all differences between recall values we therefore insert a recall value
  # of 0.0 at the beginning of the a recall list.
  #
  reca_list = [0.0]
  prec_list = []
  
  tp = 0
  fn = all_m
  fp = 0

  # Calculate precision and recall for all confusion matrices
  #
  for (class_score, class_label) in S_sorted:
    if (class_label == 'm'):
      tp += 1
      fn -= 1
    else:
      fp += 1

    prec_list.append(prec_funct(tp, fn, fp))
    reca_list.append(reca_funct(tp, fn, fp))
 
  auc_pr = 0

  for (i, prec) in enumerate(prec_list):
    reca_diff = reca_list[i+1] - reca_list[i]
    assert reca_diff >= 0, reca_diff

    auc_pr += prec * reca_diff

  assert (auc_pr >= 0.0) and (auc_pr <= 1.0), auc_pr

  # For checking we do the correct AUC-PR calculation
  #
  import sklearn.metrics
  y_true =  []
  y_score = []
  for (s,c) in S_list:
    y_score.append(s)
    if (c == 'm'):
      y_true.append(1)
    else:
      y_true.append(0)

  sklearn_auc_pr = sklearn.metrics.average_precision_score(y_true, y_score)

  if (abs(auc_pr - sklearn_auc_pr) > 0.00001):
    print('  *** Warning: sklearn AUC-PR: %.6f, our AUC-PR: %.6f ***' % \
          (sklearn_auc_pr, auc_pr))
    print('  *** Warning: Different AUC-PR values, differences is %.6f ***' \
          % (auc_pr - sklearn_auc_pr))
    print()

  return auc_pr

# -----------------------------------------------------------------------------
# Test the data set generation functions

def count_class_labels(S_list):
  num_m, num_n = 0,0
  for (s, c) in S_list:
    if (c == 'm'): num_m += 1
    elif (c == 'n'): num_n += 1
  return num_m, num_n

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

for (num_m, num_n) in [(100,400),(400,100),(100,100),(400,400),(1000,100000)]: 

  S_list = perfect_classification(num_m, num_n)
  assert len(S_list) == num_m + num_n
  m_in_S, n_in_S = count_class_labels(S_list)
  assert num_m == m_in_S
  assert num_n == n_in_S

  S_list = total_random(num_m, num_n)
  assert len(S_list) == num_m + num_n
  m_in_S, n_in_S = count_class_labels(S_list)
  assert num_m == m_in_S
  assert num_n == n_in_S

  S_list = excellent_classification(num_m, num_n)
  assert len(S_list) == num_m + num_n
  m_in_S, n_in_S = count_class_labels(S_list)
  assert num_m == m_in_S
  assert num_n == n_in_S

# -----------------------------------------------------------------------------
# Function to load a CSV file with results

def load_res_file(res_file_name):
  """Load the given file and return a list S in the same format as the lists
     generated by functions in this module.
     Also return the number of matches and non-matches identified in the file.
  """

  if (res_file_name.lower().endswith('.csv.gz')):
    in_file = gzip.open(res_file_name, 'rt')
  elif (res_file_name.lower().endswith('.csv')):
    in_file = open(res_file_name, 'rt')
  else:
    raise(Exception)

  rec_id_pair_set = set()  # Ensure each record pair only occurs once
  
  S_list = []
  s_set = set()

  num_m, num_n = 0, 0

  for line in in_file:
    line_list = line.strip().split(',')
    assert len(line_list) == 3, line
    rec_id_pair, class_score, class_label = line_list

    if (rec_id_pair in rec_id_pair_set):
      print('  *** Warning: Record pair %s occurs more than once ' % \
            (rec_id_pair) + 'in file:', res_file_name)
    rec_id_pair_set.add(rec_id_pair)

    class_score = float(class_score.strip())
    assert (class_score >= 0.0) and (class_score <= 1.0), class_score

    s_set.add(class_score)

    if (class_label.strip() == '1'):
      class_label = 'm'
      num_m += 1
    elif (class_label.strip() == '0'):
      class_label = 'n'
      num_n += 1
    else:
      raise(Exception)
    S_list.append((class_score, class_label))

  print()
  print('  Read %d classification scores and class labels from file: %s' % \
        (len(S_list), res_file_name.split('/')[-1]))
  print('    Identified %d matches and %d non-matches' % (num_m, num_n))
  class_ratio = float(num_n) / num_m
  print('      Class ratio: 1 : %.3f' % (class_ratio))
  print('      Number of unique classification scores: %d' % (len(s_set)))
  print()

  return S_list, num_m, num_n

# =============================================================================
# Start the main program

if (len(sys.argv) == 3):
  num_m = int(sys.argv[1])
  num_n = int(sys.argv[2])
  assert (num_m > 0) and (num_n > 0)  # We need both classes to be non-empty

  class_ratio = round(float(num_m) / float(num_m + num_n), 2)
  if (class_ratio == 0.5):
    class_ratio_str = '(1:1)'
  elif (class_ratio == 0.33):
    class_ratio_str = '(1:2)'
  elif (class_ratio == 0.2):
    class_ratio_str = '(1:4)'
  elif (class_ratio == 0.17):
    class_ratio_str = '(1:5)'
    
  # Specify the data sets to be evaluated, mark these are to be generated
  # For each we have a name, the function to be used to generated the data
  # set,  and the number of times to generate such a data set (due to some
  # being generated involving randomness)
  #
  data_set_list = [('gen', 'Perfect '+class_ratio_str,
                    perfect_classification, 1),
                   ('gen', 'Excellent '+class_ratio_str,
                    excellent_classification, 100),
                   ('gen', 'Good '+class_ratio_str,
                    good_classification, 100),
                   ('gen', 'Random '+class_ratio_str,
                    total_random, 1000)]

  # Print the header line for the result summary lines
  #
  print('### Data set,num_m,num_n,num_uniq_s,perc_uniq,perc_m,P t=0.05,' + \
        'R t=0.5,F1 t=0.5,F* t=0.5,P opt (t),R opt (t),F1 opt (t),' + \
        'F* opt (t),AUC-PR,CRL-0.05 (t_l/t_u),CRL-0.1 (t_l/t_u),' + \
        'CRL-0.2 (t_l/t_u),CRL-0.3 (t_l/t_u)')
  print()
  
elif (len(sys.argv) == 2):
  res_file_name = sys.argv[1]

  # Specify the data set to be loaded from file, here e only need to specify
  # it to be a data set from file and its file name
  #
  data_set_list = [('file', res_file_name)]
  
# -----------------------------------------------------------------------------
# Loop over the different data sets
#
for data_set_tuple in data_set_list:

  print('-'*80)

  # Initialise lists to hold results
  #
  P_res_list =      []  # Results for 0.5 classification threshold
  R_res_list =      []
  F_meas_res_list = []
  F_star_res_list = []

  P_opt_list =      []  # Results for optimal classification threshold
  R_opt_list =      []
  F_meas_opt_list = []
  F_star_opt_list = []

  P_opt_thres_list =      []  # And corresponding classification thresholds
  R_opt_thres_list =      []
  F_meas_opt_thres_list = []
  F_star_opt_thres_list = []

  AUC_PR_list = []  # Area under the precision recall curve result list

  CRL_05_res_list = []  # CRL-0.05
  CRL_1_res_list =  []  # CRL-0.1
  CRL_2_res_list =  []  # CRL-0.2
  CRL_3_res_list =  []  # CRL-0.3

  CRL_05_low_thres_list = []  # Corresponding upper and lower thresholds
  CRL_05_upp_thres_list = []
  CRL_1_low_thres_list =  []
  CRL_1_upp_thres_list =  []
  CRL_2_low_thres_list =  []
  CRL_2_upp_thres_list =  []
  CRL_3_low_thres_list =  []
  CRL_3_upp_thres_list =  []

  # Keep all generated or loaded score lists in one list
  #
  all_S_list = []
  
  if (data_set_tuple[0] == 'file'):
    res_file_name = data_set_tuple[1]
    data_set_name = res_file_name.split('/')[-1]

    # Load the data set from file
    #
    S_list, num_m, num_n = load_res_file(res_file_name)
    all_S_list.append(S_list)

  else:  # Get details how to generate the data set
    data_set_name, data_gen_funct, num_iter = data_set_tuple[1:]

    print()
    print('Generating data set for classification type: %s' % (data_set_name))
    print('  Running %d iterations and generating %d matches and ' % \
          (num_iter, num_m) + '%d non-matches' % (num_n))
    print()

    # Loop over the required number of generated data sets
    #
    for iter in range(num_iter):

      S_list = data_gen_funct(num_m, num_n)
      check_m, check_n = count_class_labels(S_list)
      assert (num_m == check_m) and(num_n == check_n)

      all_S_list.append(S_list)

  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  # Loop over all lists of classification scores and class labels

  for S_list in all_S_list:
  
    tp, fn, fp, tn = get_threshold_conf_matrix(S_list)
    assert tp + fn == num_m
    assert tn + fp == num_n

    # Collect the 0.5 threshold results
    #
    P_res_list.append(crl_measure.precision(tp, fn, fp))
    R_res_list.append(crl_measure.recall(tp, fn, fp))
    F_meas_res_list.append(crl_measure.f_measure(tp, fn, fp))
    F_star_res_list.append(crl_measure.f_star(tp, fn, fp))

    # Obtain and collect the optimal threshold results (with thresholds)
    #
    P_opt, P_opt_thres = get_best_result(S_list, crl_measure.precision)
    P_opt_list.append(P_opt)
    P_opt_thres_list.append(P_opt_thres)

    R_opt, R_opt_thres = get_best_result(S_list, crl_measure.recall)
    R_opt_list.append(R_opt)
    R_opt_thres_list.append(R_opt_thres)

    F_meas_opt, F_meas_opt_thres = get_best_result(S_list,
                                                   crl_measure.f_measure)
    F_meas_opt_list.append(F_meas_opt)
    F_meas_opt_thres_list.append(F_meas_opt_thres)

    F_star_opt, F_star_opt_thres = get_best_result(S_list, crl_measure.f_star)
    F_star_opt_list.append(F_star_opt)
    F_star_opt_thres_list.append(F_star_opt_thres)

    AUC_PR_list.append(calc_auc_pr(S_list, crl_measure.precision,
                                   crl_measure.recall))

    # Obtain and collect the CRL-measure results (with thresholds)
    #
    crl05, t_lower, t_upper = crl_measure.crl_measure(0.05, S_list)
    CRL_05_res_list.append(crl05)
    CRL_05_low_thres_list.append(t_lower)
    CRL_05_upp_thres_list.append(t_upper)

    crl1, t_lower, t_upper = crl_measure.crl_measure(0.1, S_list)
    CRL_1_res_list.append(crl1)
    CRL_1_low_thres_list.append(t_lower)
    CRL_1_upp_thres_list.append(t_upper)

    crl2, t_lower, t_upper = crl_measure.crl_measure(0.2, S_list)
    CRL_2_res_list.append(crl2)
    CRL_2_low_thres_list.append(t_lower)
    CRL_2_upp_thres_list.append(t_upper)

    crl3, t_lower, t_upper = crl_measure.crl_measure(0.3, S_list)
    CRL_3_res_list.append(crl3)
    CRL_3_low_thres_list.append(t_lower)
    CRL_3_upp_thres_list.append(t_upper)

  # Calculate all averaged results
  #
  P_05 =  numpy.mean(P_res_list)
  R_05 =  numpy.mean(R_res_list)
  F1_05 = numpy.mean(F_meas_res_list)
  FS_05 = numpy.mean(F_star_res_list)

  P_opt =       numpy.mean(P_opt_list)
  P_opt_thres = numpy.mean(P_opt_thres_list)
  R_opt =       numpy.mean(R_opt_list)
  R_opt_thres = numpy.mean(R_opt_thres_list)
  F1_opt =       numpy.mean(F_meas_opt_list)
  F1_opt_thres = numpy.mean(F_meas_opt_thres_list)
  FS_opt =       numpy.mean(F_star_opt_list)
  FS_opt_thres = numpy.mean(F_star_opt_thres_list)

  auc_pr =       numpy.mean(AUC_PR_list)

  crl_05 =       numpy.mean(CRL_05_res_list)
  crl_05_t_low = numpy.mean(CRL_05_low_thres_list)
  crl_05_t_upp = numpy.mean(CRL_05_upp_thres_list)
  crl_1 =        numpy.mean(CRL_1_res_list)
  crl_1_t_low =  numpy.mean(CRL_1_low_thres_list)
  crl_1_t_upp =  numpy.mean(CRL_1_upp_thres_list)
  crl_2 =        numpy.mean(CRL_2_res_list)
  crl_2_t_low =  numpy.mean(CRL_2_low_thres_list)
  crl_2_t_upp =  numpy.mean(CRL_2_upp_thres_list)
  crl_3 =        numpy.mean(CRL_3_res_list)
  crl_3_t_low =  numpy.mean(CRL_3_low_thres_list)
  crl_3_t_upp =  numpy.mean(CRL_3_upp_thres_list)

  print('  Average precision result with 0.5 threshold: ' + \
        '%.3f (std-dev: %.3f)' % (P_05, numpy.std(P_res_list)))
  print('  Average recall result with 0.5 threshold:    ' + \
        '%.3f (std-dev: %.3f)' % (R_05, numpy.std(R_res_list)))
  print('  Average F-measure result with 0.5 threshold: ' + \
        '%.3f (std-dev: %.3f)' % (F1_05, numpy.std(F_meas_res_list)))
  print('  Average F-star result with 0.5 threshold:    ' + \
        '%.3f (std-dev: %.3f)' % (FS_05, numpy.std(F_star_res_list)))  
  print()

  print('  Average precision result with optimal threshold ' + \
        '(t=%.3f): %.3f (std-dev: %.3f)' % (P_opt_thres, P_opt,
         numpy.std(P_opt_list)))
  print('  Average recall result with optimal threshold ' + \
        '(t=%.3f):    %.3f (std-dev: %.3f)' % (R_opt_thres, R_opt,
         numpy.std(R_opt_list)))
  print('  Average F-measure result with optimal threshold ' + \
        '(t=%.3f): %.3f (std-dev: %.3f)' % (F1_opt_thres, F1_opt,
         numpy.std(F_meas_opt_list)))
  print('  Average F-star result with optimal threshold ' + \
        '(t=%.3f):    %.3f (std-dev: %.3f)' % (FS_opt_thres, FS_opt,
         numpy.std(F_star_opt_list)))
  print()

  print('  Average AUC-PR result: %.3f (std-dev: %.3f)' % \
        (auc_pr, numpy.std(AUC_PR_list)))
  print()
  
  print('  Average CRL-0.05 result (t_l=%.3f, t_u=%.3f): %.3f (std-dev: %.3f)' \
        % (crl_05_t_low, crl_05_t_upp, crl_05, numpy.std(CRL_05_res_list)))
  print('  Average CRL-0.1 result  (t_l=%.3f, t_u=%.3f): %.3f (std-dev: %.3f)' \
        % (crl_1_t_low, crl_1_t_upp, crl_1, numpy.std(CRL_1_res_list)))
  print('  Average CRL-0.2 result  (t_l=%.3f, t_u=%.3f): %.3f (std-dev: %.3f)' \
        % (crl_2_t_low, crl_2_t_upp, crl_2, numpy.std(CRL_2_res_list)))
  print('  Average CRL-0.3 result  (t_l=%.3f, t_u=%.3f): %.3f (std-dev: %.3f)' \
        % (crl_3_t_low, crl_3_t_upp, crl_3, numpy.std(CRL_3_res_list)))
  print()

  # Generate a single result summary line for this data set
  #
  perc_m =   100.0*float(num_m) / (num_m + num_n)
  ci_ratio = float(num_m) / num_n

  assert num_m + num_n == len(S_list)

  uniq_s_set = set()
  for (s_ij, c_ij) in S_list:
    uniq_s_set.add(s_ij)
  num_uniq_s = len(uniq_s_set)
  perc_uniq = 100.0*float(num_uniq_s) / len(S_list)

  num_pair_per_thres = float(num_m + num_n) / num_uniq_s
  
  res_line = '%s,%d,%d,%d,%.2f,%.2f,%.3f,%.3f,%.3f,%.3f,' % (data_set_name, \
             num_m, num_n, num_uniq_s, perc_uniq, perc_m,P_05, R_05, F1_05, \
             FS_05)
  res_line += '%.3f (t=%.3f),%.3f (t=%.3f),%.3f (t=%.3f),%.3f (t=%.3f),' % \
              (P_opt, P_opt_thres, R_opt, R_opt_thres, F1_opt, F1_opt_thres,
               FS_opt, FS_opt_thres)
  res_line += '%.3f, %.3f (t_l=%.3f/t_u=%.3f),%.3f (t_l=%.3f/t_u=%.3f),' % \
              (auc_pr, crl_05, crl_05_t_low, crl_05_t_upp, crl_1, crl_1_t_low,
               crl_1_t_upp)
  res_line += '%.3f (t_l=%.3f/t_u=%.3f),%.3f (t_l=%.3f/t_u=%.3f)' % \
              (crl_2, crl_2_t_low, crl_2_t_upp, crl_3, crl_3_t_low,
               crl_3_t_upp)

  print('### '+res_line)
  print()

  # Replace real data set file names with names for paper table
  #
  ds_name_dict = \
    {'mlp_abt-buy.csv.gz':'Abt-Buy / MLP',
     'mlp_amazon-google.csv.gz':'Ama-Goog / MLP',
     'mlp_cora.csv.gz':'Cora / MLP',
     'mlp_euro.csv.gz':'Euro / MLP',
     'mlp_walmart-amazon.csv.gz':'Wal-Ama / MLP',
     #
     'random_forest_abt-buy.csv.gz':'Abt-Buy / RaFo',
     'random_forest_amazon-google.csv.gz':'Ama-Goog / RaFo',
     'random_forest_cora.csv.gz':'Cora / RaFo',
     'random_forest_euro.csv.gz':'Euro / RaFo',
     'random_forest_walmart-amazon.csv.gz':'Wal-Ama / RaFo',
     #
     'svm_linear_abt-buy.csv.gz':'Abt-Buy / L-SVM',
     'svm_linear_amazon-google.csv.gz':'Ama-Goog / L-SVM',
     'svm_linear_cora.csv.gz':'Cora / L-SVM',
     'svm_linear_euro.csv.gz':'Euro / L-SVM',
     'svm_linear_walmart-amazon.csv.gz':'Wal-Ama / L-SVM',
     #
     'svm_rbf_abt-buy.csv.gz':'Abt-Buy / R-SVM',
     'svm_rbf_amazon-google.csv.gz':'Ama-Goog / R-SVM',
     'svm_rbf_cora.csv.gz':'Cora / R-SVM',
     'svm_rbf_euro.csv.gz':'Euro / R-SVM',
     'svm_rbf_walmart-amazon.csv.gz':'Wal-Ama / R-SVM'
    }

  if (data_set_name in ds_name_dict):
    table_name = ds_name_dict[data_set_name]
  else:
    table_name = data_set_name

#  # Output for a Latex table (data set name, class imbalance ratio, optimal
#  # P, R, F*, and CRL-0.1, CRL-0.2, and CRL-0.3
#  #
#  print('Latex: %s & %.2f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f' % \
#        (table_name, ci_ratio, P_opt, R_opt, FS_opt, auc_pr, crl_1, crl_2,
#         crl_3) + ' \\\\')
#  print()

  # Output for Latex table with data set and classifier details
  # (data set name, M, N, |S_set|, (M+N)/|S_set|
  #
  latex_str1 = 'Latex: \\emph{%s} & %d & %d & %d & %.2f \\\\' % \
               (table_name, num_m, num_n, num_uniq_s, num_pair_per_thres)
  latex_str1 = latex_str1.replace('.0 ', '.00 ')
  print('Dataset table:', latex_str1)
  print()

  # Output for Latex table with classification results (data set name,
  # class_ratio, optimal P, R, F*, AUC, CRL-0.1, CRL-0.2, CRL-0.3)
  #
  latex_str2 = 'Latex: \\emph{' + \
    '%s} & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f' % \
    (table_name, ci_ratio, P_opt, R_opt, FS_opt, auc_pr, crl_1, crl_2,
     crl_3) + ' \\\\'
  latex_str2 = latex_str2.replace('.0 ', '.00 ')

  # Output a second line with the corresponding threshold values
  #
  latex_str3 = 'Latex:     & & %.3f & %.3f & %.3f & & ' % \
               (P_opt_thres, R_opt_thres, F1_opt_thres) + \
               '(%.3f,%.3f) & (%.3f,%.3f) & (%.3f,%.3f) \\\\' % \
               (crl_1_t_low, crl_1_t_upp, crl_2_t_low, crl_2_t_upp,
                crl_3_t_low, crl_3_t_upp)
  latex_str3 = latex_str3.replace('.0 ', '.00 ')
  
  print('Result table:', latex_str2)
  print('Result table:', latex_str3)
  print()
  
  # Now calculate the CRL-measure on different epsilon and print all results
  #
  print('@@@ data set, eps, crl_result, t_lower, t_upper, avr_f_star')

  if (data_set_tuple[0] == 'file'):
    assert len(all_S_list) == 1, len(all_S_list)
    for eps in eps_list:
      crl_res, crl_t_lower, crl_t_upper = crl_measure.crl_measure(eps, S_list)
      if (crl_t_upper > crl_t_lower):
        avr_f_star = min(1.0, crl_res / (crl_t_upper - crl_t_lower))
      else:
        avr_f_star = 0.0
      print('@@@ %s, %.3f, %.3f, %.3f, %.3f, %.3f' % \
            (table_name, eps, crl_res, crl_t_lower, crl_t_upper, avr_f_star))
  
  else:
    assert data_set_tuple[0] == 'gen'
    table_name = data_set_tuple[1]
    num_iter =   data_set_tuple[-1]
    assert len(all_S_list) == num_iter

    for eps in eps_list:
      crl_eps_res_list =     []
      crl_t_lower_res_list = []
      crl_t_upper_res_list = []

      for S_list in all_S_list:
        crl_res, crl_t_lower, crl_t_upper = crl_measure.crl_measure(eps, S_list)
        crl_eps_res_list.append(crl_res)
        crl_t_lower_res_list.append(crl_t_lower)
        crl_t_upper_res_list.append(crl_t_upper)
      crl_res = sum(crl_eps_res_list) / num_iter
      crl_t_lower = sum(crl_t_lower_res_list) / num_iter
      crl_t_upper = sum(crl_t_upper_res_list) / num_iter

      if (crl_t_upper > crl_t_lower):
        avr_f_star = min(1.0, crl_res / (crl_t_upper - crl_t_lower))
      else:
        avr_f_star = 0.0
      print('@@@ %s, %.3f, %.3f, %.3f, %.3f, %.3f' % \
            (table_name, eps, crl_res, crl_t_lower, crl_t_upper, avr_f_star))

  print()
