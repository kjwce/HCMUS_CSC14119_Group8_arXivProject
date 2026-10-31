from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.m2.gather import find_root, flatten, remove_iffalse, strip_comments


def write(directory: Path, name: str, text: str, encoding: str = "utf-8") -> None:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding=encoding)


class StripCommentsTests(unittest.TestCase):
    def test_escaped_percent_is_kept(self) -> None:
        self.assertEqual(strip_comments("50\\% of runs % hidden\nnext"), "50\\% of runs next")

    def test_double_backslash_before_percent_starts_comment(self) -> None:
        self.assertEqual(strip_comments("a\\\\% hidden\nb"), "a\\\\b")

    def test_comment_line_keeps_paragraph_break(self) -> None:
        self.assertEqual(strip_comments("para one%\n\npara two"), "para one\n\npara two")

    def test_verbatim_and_url_are_not_comments(self) -> None:
        text = "\\url{http://a.org/x%20y} and \\verb|50%|\n\\begin{verbatim}\n100% raw\n\\end{verbatim}\n"
        self.assertEqual(strip_comments(text), text)

    def test_iffalse_block_removed_with_nested_conditionals(self) -> None:
        text = "keep \\iffalse drop \\ifx\\a\\b x\\fi drop \\fi keep2 \\iff"
        self.assertEqual(remove_iffalse(text), "keep  keep2 \\iff")

    def test_iffalse_else_branch_is_kept(self) -> None:
        self.assertEqual(remove_iffalse("\\iffalse old\\else new\\fi"), " new")


class FlattenTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name) / "2605-00001" / "v1"
        self.dir.mkdir(parents=True)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_transitive_includes_and_commented_include(self) -> None:
        write(self.dir, "main.tex", "\\documentclass{article}\n\\begin{document}\n"
              "\\input{sections/intro}\n% \\input{old}\n\\include{appendix}\n\\end{document}\n")
        write(self.dir, "sections/intro.tex", "Intro text. \\input{sections/detail.tex}")
        write(self.dir, "sections/detail.tex", "Detail text.")
        write(self.dir, "appendix.tex", "Appendix text.")
        write(self.dir, "old.tex", "Old text.")
        result = flatten(self.dir)
        self.assertEqual(result.root, "main.tex")
        self.assertIn("Intro text. Detail text.", result.text)
        self.assertIn("Appendix text.", result.text)
        self.assertNotIn("Old text.", result.text)
        self.assertEqual(result.unused_files, ["old.tex"])
        self.assertEqual(
            result.used_files,
            ["main.tex", "sections/intro.tex", "sections/detail.tex", "appendix.tex"],
        )

    def test_cycle_and_missing_file_do_not_crash(self) -> None:
        write(self.dir, "main.tex", "\\documentclass{article}\\begin{document}\\input{a}\\input{gone}\\end{document}")
        write(self.dir, "a.tex", "A \\input{b}")
        write(self.dir, "b.tex", "B \\input{a}")
        result = flatten(self.dir)
        self.assertIn("A B", result.text)
        self.assertEqual(result.missing_files, ["gone"])
        self.assertEqual(result.cyclic_includes, ["a.tex"])

    def test_root_prefers_main_over_supplement(self) -> None:
        write(self.dir, "SI.tex", "\\documentclass{article}\\begin{document}" + "S" * 5000 + "\\end{document}")
        write(self.dir, "main.tex", "\\documentclass{article}\\begin{document}Main\\end{document}")
        self.assertEqual(find_root(self.dir).name, "main.tex")

    def test_subfile_body_only_and_import(self) -> None:
        write(self.dir, "main.tex", "\\documentclass{article}\\usepackage{subfiles}\\begin{document}"
              "\\subfile{chap/one}\\import{parts/}{two}\\end{document}")
        write(self.dir, "chap/one.tex", "\\documentclass[../main]{subfiles}\\begin{document}Chapter one.\\end{document}")
        write(self.dir, "parts/two.tex", "Part two \\input{three}")
        write(self.dir, "parts/three.tex", "and three.")
        result = flatten(self.dir)
        self.assertEqual(result.root, "main.tex")
        self.assertIn("Chapter one.", result.text)
        self.assertNotIn("[../main]", result.text)
        self.assertIn("Part two and three.", result.text)

    def test_latin1_file_and_plain_tex_input(self) -> None:
        write(self.dir, "paper.tex", "\\input macros\n\\documentclass{article}\\begin{document}Caf\xe9\\end{document}",
              encoding="latin-1")
        write(self.dir, "macros.tex", "\\def\\x{1}")
        result = flatten(self.dir)
        self.assertIn("Café", result.text)
        self.assertIn("\\def\\x{1}", result.text)
        self.assertEqual(result.non_utf8_files, ["paper.tex"])

    def test_includeonly_skips_other_includes(self) -> None:
        write(self.dir, "main.tex", "\\documentclass{article}\\includeonly{a}\\begin{document}"
              "\\include{a}\\include{b}\\end{document}")
        write(self.dir, "a.tex", "Alpha.")
        write(self.dir, "b.tex", "Beta.")
        result = flatten(self.dir)
        self.assertIn("Alpha.", result.text)
        self.assertNotIn("Beta.", result.text)

    def test_empty_version_directory(self) -> None:
        result = flatten(self.dir)
        self.assertIsNone(result.root)
        self.assertEqual(result.text, "")


if __name__ == "__main__":
    unittest.main()
