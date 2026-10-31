from __future__ import annotations

import unittest

from src.m2.normalize import clean, cite_keys, normalize_math, rename_cites, split_sentences


class SplitSentencesTests(unittest.TestCase):
    def test_abbreviations_decimals_and_citations(self) -> None:
        text = "We follow Vaswani et al. \\cite{a,b} in Fig. 1. The value is 3.14 here, i.e. small. Done!"
        self.assertEqual(
            split_sentences(text),
            ["We follow Vaswani et al. \\cite{a,b} in Fig. 1.", "The value is 3.14 here, i.e. small.", "Done!"],
        )

    def test_math_url_and_footnote_are_not_split(self) -> None:
        text = (
            "Let $x = 1. Y$ hold and \\(a. B\\) too. See \\url{http://x.org/a. B} now. "
            "End.\\footnote{A note. Two.} Next one."
        )
        self.assertEqual(
            split_sentences(text),
            [
                "Let $x = 1. Y$ hold and \\(a. B\\) too.",
                "See \\url{http://x.org/a. B} now.",
                "End.\\footnote{A note. Two.}",
                "Next one.",
            ],
        )

    def test_bare_url_and_initials(self) -> None:
        self.assertEqual(
            split_sentences("Code at https://github.com/a/b.git. J. Smith wrote it."),
            ["Code at https://github.com/a/b.git.", "J. Smith wrote it."],
        )

    def test_math_environment_and_paragraph_break(self) -> None:
        text = "Consider \\begin{math}a. B\\end{math} here\n\nNew paragraph. Sec. 2 follows."
        self.assertEqual(
            split_sentences(text),
            ["Consider \\begin{math}a. B\\end{math} here", "New paragraph.", "Sec. 2 follows."],
        )

    def test_ambiguous_abbreviation_only_before_numbers(self) -> None:
        self.assertEqual(
            split_sentences("We built an app. It works. See No. 5 and Ref. \\cite{x}."),
            ["We built an app.", "It works.", "See No. 5 and Ref. \\cite{x}."],
        )

    def test_run_in_bold_heading(self) -> None:
        self.assertEqual(
            split_sentences("\\textbf{Contributions.} We show two results."),
            ["\\textbf{Contributions.}", "We show two results."],
        )


class NormalizeMathTests(unittest.TestCase):
    def test_inline_forms(self) -> None:
        self.assertEqual(normalize_math("a \\(x\\) b \\begin{math}y\\end{math}"), "a $x$ b $y$")

    def test_block_forms(self) -> None:
        self.assertEqual(normalize_math("\\[ a=b \\]"), "\\begin{equation}a=b\\end{equation}")
        self.assertEqual(normalize_math("$$ c $$"), "\\begin{equation}c\\end{equation}")
        for env in ("equation*", "align", "align*", "gather*", "eqnarray", "multline*", "displaymath"):
            with self.subTest(env=env):
                self.assertEqual(
                    normalize_math(f"\\begin{{{env}}} x &= y \\end{{{env}}}"),
                    "\\begin{equation}x &= y\\end{equation}",
                )
        self.assertEqual(
            normalize_math("\\begin{alignat}{2} x \\end{alignat}"), "\\begin{equation}x\\end{equation}"
        )

    def test_line_break_with_spacing_is_not_display_math(self) -> None:
        self.assertEqual(
            normalize_math("\\begin{align} a \\\\[2pt] b \\end{align}"),
            "\\begin{equation}a \\\\[2pt] b\\end{equation}",
        )

    def test_inner_structures_and_escaped_dollar_kept(self) -> None:
        text = "\\begin{equation}\\begin{aligned} a \\end{aligned}\\end{equation} costs \\$5"
        self.assertEqual(normalize_math(text), text)


class CleanTests(unittest.TestCase):
    def test_figure_formatting_removed(self) -> None:
        latex = (
            "\\begin{figure*}[htbp] \\centering \\small \\includegraphics[width=\\linewidth]{f.png}"
            " \\vspace{-2mm} \\caption{A \\textbf{bold} cap~\\ref{x}.\\label{fig:x}} \\end{figure*}"
        )
        self.assertEqual(
            clean(latex),
            "\\begin{figure*} \\includegraphics[width=\\linewidth]{f.png} \\caption{A bold cap \\ref{x}.} \\end{figure*}",
        )

    def test_table_rules_removed_structure_kept(self) -> None:
        latex = "\\begin{tabular}{lc}\\toprule A & B \\\\ \\midrule \\cmidrule(lr){1-2} 1 & 2 \\\\ \\bottomrule\\end{tabular}"
        self.assertEqual(clean(latex), "\\begin{tabular}{lc} A & B \\\\ 1 & 2 \\\\ \\end{tabular}")

    def test_equation_label_and_numbering_removed(self) -> None:
        self.assertEqual(
            clean(normalize_math("\\begin{equation*}\\label{eq:1}\n x = y \\nonumber\n\\end{equation*}")),
            "\\begin{equation} x = y \\end{equation}",
        )

    def test_whitespace_and_url_tilde(self) -> None:
        self.assertEqual(clean("  a\n\n b~c \\url{http://u.edu/~me}  "), "a b c \\url{http://u.edu/~me}")

    def test_commands_with_letter_suffix_untouched(self) -> None:
        self.assertEqual(clean("\\smallskip\\item \\itemize \\paragraph{P}"), "\\item \\itemize \\paragraph{P}")


class RenameCitesTests(unittest.TestCase):
    def test_all_cite_variants(self) -> None:
        mapping = {"vaswani2017": "vit_transformer"}
        text = (
            "\\citep[see][p.~2]{vaswani2017, other} \\cite{vaswani2017,vit_transformer} "
            "\\citeauthor*{vaswani2017} \\parencites{vaswani2017}{y} \\Citet{z}"
        )
        self.assertEqual(
            rename_cites(text, mapping),
            "\\citep[see][p.~2]{vit_transformer,other} \\cite{vit_transformer} "
            "\\citeauthor*{vit_transformer} \\parencites{vit_transformer}{y} \\Citet{z}",
        )

    def test_cite_keys(self) -> None:
        self.assertEqual(cite_keys("\\cite{a, b} \\citet[x]{c} \\nocite{d} \\cites{e}{f}"), list("abcdef"))


if __name__ == "__main__":
    unittest.main()
