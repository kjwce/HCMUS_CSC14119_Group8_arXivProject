from __future__ import annotations

import re
import unittest

from src.m2.hierarchy import build_tree, strip_comments


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


class HierarchyTests(unittest.TestCase):
    def test_comments_headings_and_reference_exclusion(self) -> None:
        latex = r"""
        \documentclass{article}
        \begin{document}
        % \section{Commented out}
        \section{Introduction}
        First sentence. Second sentence.
        \subsection{Details}
        Details remain. \%
        \section*{Acknowledgements}
        Thanks to everyone.
        \appendix
        \section{Extra proof}
        Appendix sentence.
        \section*{References}
        This must disappear.
        \end{document}
        """
        root = build_tree(latex, split_sentences)
        kinds_and_content = [(node.kind, node.content) for node, _ in root.walk()]
        self.assertIn(("section", "Introduction"), kinds_and_content)
        self.assertIn(("subsection", "Details"), kinds_and_content)
        self.assertIn(("section", "Acknowledgements"), kinds_and_content)
        self.assertIn(("section", "Extra proof"), kinds_and_content)
        self.assertNotIn(("section", "References"), kinds_and_content)
        self.assertFalse(any("must disappear" in content for _, content in kinds_and_content))

    def test_formula_figure_and_nested_list(self) -> None:
        latex = r"""
        \begin{document}
        \section{Method}
        Before formula.
        \begin{align}a &= b\end{align}
        \begin{figure}[ht]\caption{A figure}\end{figure}
        \begin{itemize}
          \item First item.
          \item Second item.
            \begin{enumerate}\item Nested item.\end{enumerate}
        \end{itemize}
        \end{document}
        """
        root = build_tree(latex, split_sentences)
        kinds = [node.kind for node, _ in root.walk()]
        self.assertEqual(kinds.count("formula"), 1)
        self.assertEqual(kinds.count("figure"), 1)
        self.assertEqual(kinds.count("list"), 2)
        self.assertEqual(kinds.count("item"), 3)

    def test_standalone_table_environments_are_figures(self) -> None:
        latex = r"""
        \begin{document}
        \begin{deluxetable*}{lc}
        A & B \\
        \end{deluxetable*}
        \begin{sidewaystable*}
        \begin{tabular}{lc}C & D\end{tabular}
        \end{sidewaystable*}
        \begin{center}
        \begin{tabularx}{\textwidth}{lc}E & F\end{tabularx}
        \end{center}
        \end{document}
        """
        root = build_tree(latex, split_sentences)
        figures = [node.content for node, _ in root.walk() if node.kind == "figure"]
        self.assertEqual(len(figures), 3)
        self.assertTrue(any("deluxetable*" in content for content in figures))
        self.assertTrue(any("sidewaystable*" in content for content in figures))
        self.assertTrue(any("tabularx" in content for content in figures))

    def test_escaped_percent_is_not_a_comment(self) -> None:
        self.assertEqual(strip_comments("100\\% kept % removed\nnext"), "100\\% kept \nnext")

    def test_front_matter_abstract_and_formula_connectors(self) -> None:
        latex = r"""
        \begin{document}
        \title{Leaked title}
        \author[0000-0000]{Leaked author}[orcid=0000-0000]
        \affiliation{Leaked affiliation}
        \cormark[1]
        \cortext[1]{Leaked corresponding-author note}
        \date{\today}
        \maketitle
        \begin{abstract}
        Abstract first sentence. Abstract second sentence.
        \end{abstract}
        \begin{keywords}Leaked keyword\end{keywords}
        \section{Body}
        A complete sentence.
        \begin{equation}a=b\end{equation}
        with
        \begin{equation}c=d\end{equation}
        Thus:
        \begin{equation}e=f\end{equation}
        \begin{proof}Proof sentence.\end{proof}
        \end{document}
        """
        root = build_tree(latex, split_sentences)
        nodes = [(node.kind, node.content) for node, _ in root.walk()]
        contents = [content for _, content in nodes]
        abstracts = [node for node, _ in root.walk() if node.kind == "abstract"]

        self.assertEqual(len(abstracts), 1)
        self.assertEqual(
            [child.content for child in abstracts[0].children],
            ["Abstract first sentence.", "Abstract second sentence."],
        )
        self.assertFalse(any("Leaked" in content or "abstract}" in content for content in contents))
        self.assertNotIn("with", contents)
        self.assertIn("Thus:", contents)
        self.assertIn("Proof sentence.", contents)
        self.assertFalse(any("\\begin{proof}" in content or "\\end{proof}" in content for content in contents))


if __name__ == "__main__":
    unittest.main()

