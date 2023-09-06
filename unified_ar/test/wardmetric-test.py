from unified_ar.wardmetrics.core_methods import eval_events, eval_segments
from unified_ar.wardmetrics.utils import *
from unified_ar.wardmetrics.visualisations import *

from unified_ar.metric.MyMetric import testMyMetric

ground_truth_test = [
    (40, 60),
    (70, 75),
    (90, 100),
    (125, 135),
    (150, 157),
    (187, 220),
]

detection_test = [
    (10, 20),
    (45, 52),
    (65, 80),
    (120, 180),
    (195, 200),
    (207, 213),
]

ground_truth_test = [
    (40, 60),
    (73, 75),
    (90, 100),
    (125, 135),
    (150, 157),
    (190, 215),
    (220, 230),
    (235, 250),
    (275, 292),
    (340, 368),
]


detection_test = [
    (10, 20),
    (45, 52),
    (70, 80),
    (120, 180),
    (195, 200),
    (207, 213),
    (221, 237),
    (239, 243),
    (245, 250),
]

testMyMetric(ground_truth_test, detection_test)
eval_start = 2
eval_end = 241

# Calculate segment results:
twoset_results, segments_with_scores, segment_counts, normed_segment_counts = eval_segments(ground_truth_test, detection_test, eval_start, eval_end)

# Print results:
print_detailed_segment_results(segment_counts)
print_detailed_segment_results(normed_segment_counts)
print_twoset_segment_metrics(twoset_results)

# Access segment results in other formats:
print("\nAbsolute values:")
print("----------------")
print(detailed_segment_results_to_list(segment_counts))  # segment scores as basic python list
print(detailed_segment_results_to_string(segment_counts))  # segment scores as string line
print(detailed_segment_results_to_string(segment_counts, separator=";", prefix="(", suffix=")\n"))  # segment scores as string line

print("Normed values:")
print("--------------")
print(detailed_segment_results_to_list(normed_segment_counts))  # segment scores as basic python list
print(detailed_segment_results_to_string(normed_segment_counts))  # segment scores as string line
print(detailed_segment_results_to_string(normed_segment_counts, separator=";", prefix="(", suffix=")\n"))  # segment scores as string line

# Access segment metrics in other formats:
print("2SET metrics:")
print("-------------")
print(twoset_segment_metrics_to_list(twoset_results))  # twoset_results as basic python list
print(twoset_segment_metrics_to_string(twoset_results))  # twoset_results as string line
print(twoset_segment_metrics_to_string(twoset_results, separator=";", prefix="(", suffix=")\n"))  # twoset_results as string line

# Visualisations:
plot_events_with_segment_scores(segments_with_scores, ground_truth_test, detection_test)
plot_segment_counts(segment_counts)
plot_twoset_metrics(twoset_results)


# Run event-based evaluation:
gt_event_scores, det_event_scores, detailed_scores, standard_scores = eval_events(ground_truth_test, detection_test)

# Print results:
print_standard_event_metrics(standard_scores)
print_detailed_event_metrics(detailed_scores)

# Access results in other formats:
print(standard_event_metrics_to_list(standard_scores))  # standard scores as basic python list, order: p, r, p_w, r_w
print(standard_event_metrics_to_string(standard_scores))  # standard scores as string line, order: p, r, p_w, r_w)
print(standard_event_metrics_to_string(standard_scores, separator=";", prefix="(", suffix=")\n"))  # standard scores as string line, order: p, r, p_w, r_w

print(detailed_event_metrics_to_list(detailed_scores))  # detailed scores as basic python list
print(detailed_event_metrics_to_string(detailed_scores))  # detailed scores as string line
print(detailed_event_metrics_to_string(detailed_scores, separator=";", prefix="(", suffix=")\n"))  # standard scores as string line


# Show results:
plot_events_with_event_scores(gt_event_scores, det_event_scores, ground_truth_test, detection_test, show=False)
plot_event_analysis_diagram(detailed_scores)
