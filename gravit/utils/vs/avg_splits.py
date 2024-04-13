import re


def print_results(all_eval_results):
    """
    Given a set of results from all splits, calculates avg value for each metric.
    """
    all_f1_scores = []
    all_taus = []
    all_rhos = []
    for eval in all_eval_results:
        _, f1, tau, rho = re.findall("\d+\.?\d*", eval)
        all_f1_scores.append(float(f1))
        all_taus.append(float(tau))
        all_rhos.append(float(rho))

    final_f1_score = sum(all_f1_scores) / len(all_f1_scores)
    final_tau = sum(all_taus) / len(all_taus)
    final_rho = sum(all_rhos) / len(all_rhos)

    print(f"Final average results: F1-Score = {final_f1_score:.4}, Kendall's Tau = {final_tau:.3}, Spearman's Rho = {final_rho:.3}")
