"""Build the six-page assignment report from the saved results and figures."""

import argparse
import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)


ROOT = Path(__file__).resolve().parent
WIDTH = A4[0] - 90
STYLES = getSampleStyleSheet()
STYLES.add(ParagraphStyle("Body", fontName="Times-Roman", fontSize=11,
                          leading=14, spaceAfter=8))
STYLES.add(ParagraphStyle("Heading", fontName="Times-Bold", fontSize=15,
                          leading=18, spaceBefore=4, spaceAfter=10))
STYLES.add(ParagraphStyle("Subheading", fontName="Times-Bold", fontSize=12,
                          leading=15, spaceBefore=6, spaceAfter=7))
STYLES.add(ParagraphStyle("Caption", fontName="Times-Italic", fontSize=9.5,
                          leading=12, spaceAfter=9))
STYLES.add(ParagraphStyle("ReportTitle", fontName="Times-Bold", fontSize=24,
                          leading=29, alignment=TA_CENTER, spaceAfter=10))
STYLES.add(ParagraphStyle("Author", fontName="Times-Roman", fontSize=12,
                          leading=16, alignment=TA_CENTER, spaceAfter=5))
STYLES.add(ParagraphStyle("CodeBlock", fontName="Courier", fontSize=8.5,
                          leading=12, spaceAfter=8))


def paragraph(text, style="Body"):
    return Paragraph(text, STYLES[style])


def table(rows, widths=None, small=False):
    result = Table(rows, colWidths=widths, hAlign="LEFT", repeatRows=1)
    result.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Times-Roman"),
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9 if small else 10),
        ("LEADING", (0, 0), (-1, -1), 10.5 if small else 12),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf1f4")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#657888")),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.HexColor("#657888")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f8f9")]),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2 if small else 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2 if small else 5),
    ]))
    return result


def figure(root, filename, width=WIDTH):
    image = Image(str(root / "figures" / filename))
    image.drawHeight *= width / image.drawWidth
    image.drawWidth = width
    return image


def selection_page(root, label, data, rows, page_number):
    number = "1" if label == "var1" else "2"
    elements = [paragraph(f"{page_number}. Var{number}: degree and regularization selection", "Heading")]
    elements.append(paragraph(
        f"The search evaluated every total degree from 1 to {data['max_degree']} and all 50 "
        "alpha values on the same five development folds. For each degree, the table shows "
        "the alpha with the lowest mean validation MSE. The final selection minimizes this "
        "MSE over both degree and alpha."
    ))
    elements.append(figure(root, f"{label}_degree_vs_error.png", width=460))
    elements.append(paragraph(
        f"Figure {number}. Validation MSE on a logarithmic axis. The shaded band is plus/minus "
        "one standard deviation across the five folds, not a confidence interval.", "Caption"
    ))
    content = [["Degree", "Alpha", "Mean CV MSE", "Fold SD", "Mean CV R2"]]
    for row in rows:
        content.append([str(int(row["degree"])), f"{float(row['alpha']):.6g}",
                        f"{float(row['mean_cv_mse']):.6f}", f"{float(row['std_cv_mse']):.6f}",
                        f"{float(row['mean_cv_r2']):.6f}"])
    elements.append(table(content, [54, 100, 120, 100, WIDTH - 374], small=True))
    elements.append(Spacer(1, 8))
    elements.append(paragraph(
        f"The lowest evaluated mean CV MSE is {data['selection_cv']['mse']:.6f}, at degree "
        f"{data['selected_degree']} and alpha {data['selected_alpha']:.8g}. "
        "This identifies the best pair among the evaluated configurations. The CV score "
        "also guided selection, so the separate holdout score is used for the final assessment."
    ))
    return elements


def assessment_page(root, label, data, page_number):
    number = "1" if label == "var1" else "2"
    elements = [paragraph(f"{page_number}. Var{number}: assessment and final model", "Heading")]
    elements.append(paragraph(
        f"The selected degree {data['selected_degree']} generates {data['n_polynomial_features']:,} "
        f"nonconstant polynomial terms. Ridge adds a separate intercept, giving "
        f"{data['n_polynomial_features'] + 1:,} fitted coefficients in total. "
        f"The alpha is {data['selected_alpha']:.10g}. The model fitted on 800 development rows "
        "was evaluated once on the 200 holdout rows. Degree and alpha remained fixed afterward."
    ))
    content = [["Evaluation", "Rows per evaluation", "MSE", "R2"]]
    content.append(["Selected pair, mean CV", "160 per fold",
                    f"{data['selection_cv']['mse']:.6f}", f"{data['selection_cv']['r2']:.6f}"])
    for name, key, n in [("Development training", "development_train", "800"),
                         ("Untouched holdout", "holdout", "200"),
                         ("Final model training", "final_train", "1,000")]:
        content.append([name, n, f"{data[key]['mse']:.6f}", f"{data[key]['r2']:.6f}"])
    elements.append(table(content, [175, 140, 95, WIDTH - 410]))
    elements.append(Spacer(1, 12))
    elements.append(paragraph(
        "The final model then fits all 1,000 labeled rows with the selected settings. "
        "Its training score measures how closely it fits those rows. The holdout result "
        "above belongs to the earlier 800-row model; it is not a test score for the refitted model."
    ))
    elements.append(figure(root, f"{label}_residuals.png"))
    elements.append(paragraph(
        f"Figure {int(number) + 2}. Training residuals after the final 1,000-row refit. "
        "A residual is the observed target minus the fitted target.", "Caption"
    ))
    elements.append(paragraph(
        "The residual plot checks for systematic training errors and unusually large errors. "
        "The fitted intercept makes the mean training residual approximately zero. "
        "A roughly centered histogram alone does not establish performance on unseen rows. "
        "The independent holdout result provides the direct evidence available for that purpose."
    ))
    elements.append(paragraph("Test inference", "Subheading"))
    elements.append(paragraph(
        f"The saved final pipeline transforms the {len(data['input_features'])} inputs in each "
        "test row using its training-fitted polynomial expansion and scaler, then predicts y. "
        f"The output BT2024103_pred_{label}.csv contains one y column and 1,000 rows in the "
        "original test-file order, with no index column. Hidden test targets are unavailable, "
        "so this report does not claim test MSE or test R2."
    ))
    return elements


def footer(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#d4d9dd"))
    canvas.line(45, 38, A4[0] - 45, 38)
    canvas.setFont("Times-Roman", 9)
    canvas.drawString(45, 25, "BT2024103 | Polynomial regression")
    canvas.drawRightString(A4[0] - 45, 25, str(document.page))
    canvas.restoreState()


def build_report(root):
    metrics = json.loads((root / "results" / "metrics.json").read_text(encoding="utf-8"))
    datasets = metrics["datasets"]
    protocol = metrics["protocol"]
    if protocol["holdout_used_for_selection"] or len(protocol["alpha_grid"]) != 50:
        raise ValueError("Saved results do not match the report's documented protocol")
    rows = {}
    for label in datasets:
        with (root / "results" / f"{label}_degree_selection.csv").open(newline="", encoding="utf-8") as stream:
            rows[label] = list(csv.DictReader(stream))
    story = [paragraph("Polynomial regression", "ReportTitle"),
             paragraph("Machine Learning Assignment 1", "Author"),
             paragraph("Yashwanth Reddy Pidela | Roll number BT2024103", "Author"),
             paragraph('<link href="https://github.com/TheOrangeCarXX/polynomial_regression" '
                       'color="#215b87">github.com/TheOrangeCarXX/polynomial_regression</link>', "Author"),
             Spacer(1, 13), paragraph("1. Data, model, and evaluation procedure", "Heading")]
    story.append(paragraph(
        "The assignment asks for two polynomial regression models with different input counts "
        "and degree limits. Each personalized training file contains 1,000 labeled rows. "
        "Each corresponding test file contains 1,000 rows without target values. The four "
        "input files were checked for the required column order, numeric values, and missing "
        "or infinite values before fitting."
    ))
    story.append(table([["Dataset", "Input columns", "Degrees evaluated", "Train / test rows"],
                        ["Var1", "x1 through x6", "1 through 10", "1,000 / 1,000"],
                        ["Var2", "x1 through x3", "1 through 20", "1,000 / 1,000"]],
                       [70, 140, 145, WIDTH - 355]))
    story.append(paragraph("Polynomial expansion and Ridge regression", "Subheading"))
    story.append(paragraph(
        "A total-degree polynomial of degree d contains every product of the inputs whose "
        "exponents sum to at most d. With three inputs and d = 2, the nonconstant terms are "
        "x1, x2, x3, x1<super>2</super>, x1x2, x1x3, x2<super>2</super>, x2x3, and "
        "x3<super>2</super>. A separate intercept supplies the constant term. The number of "
        "nonconstant terms for p inputs is C(p + d, d) - 1, where C denotes the binomial coefficient."
    ))
    story.append(paragraph(
        "PolynomialFeatures builds these terms. StandardScaler subtracts each term's training "
        "mean and divides by its training standard deviation. Ridge estimates the coefficients "
        "by minimizing the sum of squared prediction errors plus alpha times the sum of squared "
        "non-intercept coefficients. Alpha controls the strength of this penalty. It limits "
        "large coefficients when polynomial columns are correlated. The intercept is unpenalized."
    ))
    story.append(paragraph("Selection, holdout, and refitting", "Subheading"))
    story.append(paragraph(
        f"A shuffled split with seed {protocol['split_seed']} reserves 200 rows as a holdout and "
        f"uses 800 rows for development. Five-fold cross-validation with seed {protocol['cv_seed']} "
        "divides the development rows into five groups. Each fold fits on 640 rows and validates "
        "on the remaining 160. Every fold fits its own polynomial expansion and scaler. "
        "Validation and holdout rows never contribute to those fitted preprocessing statistics."
    ))
    story.append(paragraph(
        "The search evaluates 50 logarithmically spaced alpha values from 0.0001 to 10,000 "
        "for every allowed degree. One singular value decomposition per degree and fold reuses "
        "the Ridge solution across alphas. The degree-and-alpha pair with the lowest mean "
        "validation MSE is selected. That pair fits the 800 development rows and is assessed "
        "once on the untouched holdout. Finally, the same pair fits all 1,000 labeled rows "
        "and predicts the supplied test rows."
    ))
    story.append(paragraph(
        "MSE is the mean squared difference between actual and predicted targets, so smaller "
        "values indicate smaller errors. R2 compares squared prediction error with the target's "
        "variation about its mean. A value of 1 is a perfect fit; a value below 0 is worse "
        "than predicting that evaluation set's mean. CV R2 is the mean of the five fold scores."
    ))
    story.append(PageBreak())
    story.extend(selection_page(root, "var1", datasets["var1"], rows["var1"], 2))
    story.append(PageBreak())
    story.extend(assessment_page(root, "var1", datasets["var1"], 3))
    story.append(PageBreak())
    story.extend(selection_page(root, "var2", datasets["var2"], rows["var2"], 4))
    story.append(PageBreak())
    story.extend(assessment_page(root, "var2", datasets["var2"], 5))
    story.append(PageBreak())
    story.append(paragraph("6. Reproduction, deliverables, and interpretation", "Heading"))
    story.append(paragraph(
        "The repository contains all training and inference code, both prediction CSVs, "
        "four generated figures, full search results, final saved models, and this report's "
        "Python source. Input datasets are obtained using the download link in the assignment "
        "brief and placed together in data/. The README gives the exact four expected filenames."
    ))
    story.append(paragraph("Reproduction commands", "Subheading"))
    story.append(paragraph("python -m pip install -r requirements.txt<br/>"
                           "python ml_assignment_solution.py --data-dir data<br/>"
                           "python generate_report.py", "CodeBlock"))
    story.append(paragraph(
        "The first script reruns the full search and saves results under results/, figures "
        "under figures/, and final models under models/. The report generator reads those "
        "saved results and images. Its tables and numerical descriptions therefore come "
        "from the same run as the prediction files. Paths are command-line arguments or "
        "relative to the script location; no personal Windows path is required."
    ))
    story.append(paragraph("Inference without repeating model selection", "Subheading"))
    story.append(paragraph("python ml_assignment_solution.py --data-dir data --predict-only", "CodeBlock"))
    story.append(paragraph(
        "This command loads the supplied final pipelines and regenerates both prediction "
        "files. It requires only the two test input CSVs. Each saved pipeline contains the "
        "polynomial transform, fitted scaler, and fitted Ridge model."
    ))
    story.append(paragraph("Results and limits", "Subheading"))
    comparison = [["Dataset", "Degree", "Alpha", "Holdout MSE", "Holdout R2"]]
    for label, data in datasets.items():
        comparison.append([label.capitalize(), str(data["selected_degree"]),
                           f"{data['selected_alpha']:.6g}", f"{data['holdout']['mse']:.6f}",
                           f"{data['holdout']['r2']:.6f}"])
    story.append(table(comparison, [70, 70, 110, 130, WIDTH - 380]))
    story.append(Spacer(1, 9))
    story.append(paragraph(
        "These are the lowest-CV-MSE configurations among the degrees and alphas evaluated. "
        "They are not a claim of a globally optimal polynomial. Selection CV scores can be "
        "optimistic because the search chooses their minimum. The holdout avoids that reuse "
        "during selection, although a single 200-row split still has sampling uncertainty. "
        "The final full-data refit has no separate observed test score because the assignment "
        "withholds the test targets."
    ))
    story.append(paragraph("Recorded environment", "Subheading"))
    version_text = "; ".join(f"{escape(name)} {escape(version)}" for name, version in metrics["versions"].items())
    story.append(paragraph(version_text + ". Pinned dependencies appear in requirements.txt. "
                           "Input SHA-256 hashes and the split settings appear in results/metrics.json. "
                           "Small floating-point differences across numerical libraries or machines are possible."))
    story.append(paragraph("References", "Subheading"))
    story.append(paragraph(
        "Assignment 1 brief supplied with the datasets. scikit-learn documentation for "
        '<link href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html" color="#215b87">PolynomialFeatures</link>, '
        '<link href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html" color="#215b87">StandardScaler</link>, '
        '<link href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html" color="#215b87">Ridge</link>, and '
        '<link href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html" color="#215b87">KFold</link>.'
    ))
    destination = root / "ML_Assignment_1_Report.pdf"
    document = SimpleDocTemplate(str(destination), pagesize=A4, leftMargin=45, rightMargin=45,
                                 topMargin=43, bottomMargin=50,
                                 title="Polynomial regression - ML Assignment 1",
                                 author="Yashwanth Reddy Pidela")
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=ROOT,
                        help="Folder containing results/ and figures/, and receiving the PDF")
    args = parser.parse_args()
    print(f"Saved report to {build_report(args.results_root.resolve())}")


if __name__ == "__main__":
    main()
