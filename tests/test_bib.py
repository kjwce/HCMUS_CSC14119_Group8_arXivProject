from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from src.m2.bib import (
    Entry,
    dedup,
    extract,
    latex_to_text,
    normalize_arxiv,
    normalize_doi,
    parse_bib_text,
    parse_bibitems,
    parse_bibtex_authors,
    select_entries,
)


def item(text: str) -> Entry:
    entries = parse_bibitems(text)
    assert len(entries) == 1, entries
    return entries[0]


class FieldTests(unittest.TestCase):
    def test_latex_to_text(self) -> None:
        self.assertEqual(latex_to_text("J{\\'e}r{\\^o}me M\\\"uller \\& {GAN}s --- x"), "Jérôme Müller & GANs — x")

    def test_bibtex_authors(self) -> None:
        self.assertEqual(
            parse_bibtex_authors("Vaswani, Ashish and Noam Shazeer and {Google Brain} and others"),
            ["Ashish Vaswani", "Noam Shazeer", "Google Brain"],
        )

    def test_identifiers(self) -> None:
        self.assertEqual(normalize_doi("https://doi.org/10.1103/PhysRevA.1.2."), "10.1103/physreva.1.2")
        self.assertEqual(normalize_doi("(doi:10.1000/example)."), "10.1000/example")
        self.assertEqual(
            normalize_doi("doi:10.1002/(SICI)1099-0844(199912)17:4"),
            "10.1002/(sici)1099-0844(199912)17:4",
        )
        self.assertEqual(normalize_arxiv("arXiv preprint arXiv:1706.03762v5"), "1706.03762")
        self.assertEqual(normalize_arxiv("hep-th/9901001"), "hep-th/9901001")


class FastBibParserTests(unittest.TestCase):
    def test_strings_concatenation_quotes_and_parentheses(self) -> None:
        text = (
            '@string{jmlr = "Journal of {ML} Research"}\n'
            "@comment{ignored @article{fake, title={No}} }\n"
            '@article{a, title = "A {"}quoted{"} title", journal = jmlr # " (JMLR)",\n'
            "  author={Doe, Jane and Roe, R.}, year = 2020, month = jan, note={mail me@x.org},}\n"
            "@inproceedings(b, title={Nested {Braces} here}, booktitle={Proc})\n"
        )
        entries = parse_bib_text(text)
        self.assertEqual([entry["ID"] for entry in entries], ["a", "b"])
        a, b = entries
        self.assertEqual(a["title"], 'A {"}quoted{"} title')
        self.assertEqual(a["journal"], "Journal of {ML} Research (JMLR)")
        self.assertEqual((a["year"], a["month"], a["note"]), ("2020", "January", "mail me@x.org"))
        self.assertEqual((b["ENTRYTYPE"], b["title"]), ("inproceedings", "Nested {Braces} here"))

    def test_select_entries_keeps_strings_and_cited(self) -> None:
        text = "@string{x = {X}}\n@article{keep, title={K}}\n@article{drop, title={D}}\n"
        self.assertEqual([entry["ID"] for entry in parse_bib_text(select_entries(text, {"keep"}))], ["keep"])

    def test_malformed_file_falls_back_without_crashing(self) -> None:
        entries = parse_bib_text("@article{ok, title={Fine}}\n@article{broken, title={Unclosed\n")
        self.assertIsInstance(entries, list)


class BibitemTests(unittest.TestCase):
    def test_ieee_quoted_title(self) -> None:
        entry = item(
            "\\bibitem{GRW} G. C. Ghirardi, A. Rimini, and T. Weber, ``Unified dynamics,'' "
            "Phys. Rev. D \\textbf{34}, 470 (1986)."
        )
        self.assertEqual(entry.citation_key, "GRW")
        self.assertEqual(entry.title, "Unified dynamics")
        self.assertEqual(entry.authors, ["G. C. Ghirardi", "A. Rimini", "T. Weber"])
        self.assertEqual((entry.year, entry.journal, entry.entry_type), (1986, "Phys. Rev. D", "article"))

    def test_natbib_newblock(self) -> None:
        entry = item(
            "\\bibitem[Aziz and Monnot(2020)]{Aziz2020} Haris Aziz and J{\\'e}r{\\^o}me Monnot. "
            "\\newblock Computing pareto optimal committees. \\newblock \\emph{Autonomous Agents}, 2020. "
            "\\newblock \\doi{10.1007/s10458-020-09445-y}."
        )
        self.assertEqual(entry.authors, ["Haris Aziz", "Jérôme Monnot"])
        self.assertEqual(entry.title, "Computing pareto optimal committees")
        self.assertEqual(entry.journal, "Autonomous Agents")
        self.assertEqual(entry.doi, "10.1007/s10458-020-09445-y")

    def test_apa_author_year(self) -> None:
        entry = item(
            "\\bibitem{av} Avellaneda, M. and Stoikov, S. (2008). High-frequency trading in a limit "
            "order book. \\emph{Quantitative Finance}, 8(3), 217--224."
        )
        self.assertEqual(entry.authors, ["M. Avellaneda", "S. Stoikov"])
        self.assertEqual(entry.title, "High-frequency trading in a limit order book")
        self.assertEqual((entry.year, entry.journal), (2008, "Quantitative Finance"))

    def test_acs_semicolon_authors(self) -> None:
        entry = item(
            "\\bibitem{rong} Rong,~Z.; Malik,~R.; Ceder,~G. Materials design rules. "
            "\\emph{Chemistry of Materials} \\textbf{2015}, \\emph{27}, 6016--6021\\relax"
        )
        self.assertEqual(entry.authors, ["Z. Rong", "R. Malik", "G. Ceder"])
        self.assertEqual(entry.title, "Materials design rules")
        self.assertEqual((entry.year, entry.journal), (2015, "Chemistry of Materials"))

    def test_math_style_italic_title(self) -> None:
        entry = item(
            "\\bibitem{Ar} D.~Arapura, {\\em Geometry of cohomology support loci}, "
            "J. Algebraic Geom. \\textbf{6} (1997), no.~3, 563--597."
        )
        self.assertEqual(entry.authors, ["D. Arapura"])
        self.assertEqual(entry.title, "Geometry of cohomology support loci")
        self.assertEqual(entry.year, 1997)

    def test_initials_first_with_year_after_authors(self) -> None:
        # Regression: 2603-05580 leaked the last author into the title.
        entry = item(
            "\\bibitem{AAV} Y. Aharonov, D.Z. Albert, L. Vaidmann, 1988, How the result of a measurement "
            "can turn out to be 100, \\textit{Phys. Rev. Lett.} \\textbf{60} (1988), 1351-1354."
        )
        self.assertEqual(entry.authors, ["Y. Aharonov", "D.Z. Albert", "L. Vaidmann"])
        self.assertEqual(entry.title, "How the result of a measurement can turn out to be 100")
        self.assertEqual((entry.journal, entry.year), ("Phys. Rev. Lett", 1988))
        entry = item(
            "\\bibitem{c} Y. Aharonov, I. Sabadini, D. C. Struppa, and J. Tollaksen, 2017,\n"
            "The Mathematics of Superoscillations,\n\\textit{Memoirs of the AMS}, Vol. 247."
        )
        self.assertEqual(entry.authors, ["Y. Aharonov", "I. Sabadini", "D. C. Struppa", "J. Tollaksen"])
        self.assertEqual(entry.title, "The Mathematics of Superoscillations")

    def test_initials_first_without_commas_or_italic_venue(self) -> None:
        lee = item(
            "\\bibitem{Lee} D. G. Lee and P. J. S. G. Ferreira 2014 Direct construction of superoscillations, "
            "IEEE \\textit{ Trans. Signal Process.} 62, 3125-34."
        )
        self.assertEqual(lee.authors, ["D. G. Lee", "P. J. S. G. Ferreira"])
        self.assertEqual((lee.title, lee.journal), ("Direct construction of superoscillations", "IEEE Trans. Signal Process"))
        weier = item(
            "\\bibitem{w} K.Weierstrass, \\\"Uber continuirliche Functionen eines reellen Arguments, "
            "[On continuous functions], in: K\\\"oniglich Preu\\ss ische Akademie der Wissenschaften, "
            "Mathematische Werke (Berlin: Mayer \\& Mueller, 1895), vol. 2, pages 71--74."
        )
        self.assertEqual(weier.authors, ["K. Weierstrass"])
        self.assertEqual(weier.title, "Über continuirliche Functionen eines reellen Arguments")
        self.assertEqual(weier.journal, "Königlich Preußische Akademie der Wissenschaften")

    def test_physics_style_without_title(self) -> None:
        entry = item("\\bibitem{EM} M. R. Evans and S. N. Majumdar, Phys. Rev. Lett. \\textbf{106}, 160601 (2011).")
        self.assertEqual(entry.authors, ["M. R. Evans", "S. N. Majumdar"])
        self.assertIsNone(entry.title)
        self.assertEqual((entry.year, entry.journal), (2011, "Phys. Rev. Lett"))

    def test_revtex_bibinfo(self) -> None:
        entry = item(
            "\\bibitem[{Ab(2020)}]{rx}\\BibitemOpen \\bibfield {author} {\\bibinfo {author} {\\bibfnamefont {A.}~"
            "\\bibnamefont {Abc}}}\\bibfield {title} {\\bibinfo {title} {Quantum things},\\ }"
            "\\href {https://doi.org/10.1103/PhysRevA.1.2} {\\bibfield {journal} {\\bibinfo {journal} {Phys. Rev. A}\\ }"
            "(\\bibinfo {year} {2020})}\\BibitemShut {NoStop}"
        )
        self.assertEqual((entry.title, entry.authors, entry.year), ("Quantum things", ["A. Abc"], 2020))
        self.assertEqual((entry.journal, entry.doi), ("Phys. Rev. A", "10.1103/physreva.1.2"))

    def test_arxiv_preprint_and_access_date(self) -> None:
        entry = item(
            "\\bibitem{d} G.~Dhama. \\newblock On limits. \\newblock \\emph{arXiv preprint arXiv:2605.27557}, 2026."
        )
        self.assertEqual((entry.eprint, entry.entry_type, entry.year), ("2605.27557", "misc", 2026))
        web = item(
            "\\bibitem{w} OpenAI, ``ChatGPT,'' \\emph{OpenAI Blog}, 2022. [Online]. "
            "Available: \\url{https://openai.com/blog}. [Accessed: Oct. 2, 2025]."
        )
        self.assertEqual((web.year, web.url), (2022, "https://openai.com/blog"))

    def test_acronym_heavy_item_does_not_backtrack(self) -> None:
        # Regression: 2606-00335 hung the corpus run on such an item.
        started = time.perf_counter()
        entry = item("\\bibitem{x} A. Author, " + "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 3 + " groups")
        self.assertLess(time.perf_counter() - started, 1.0)
        self.assertEqual(entry.citation_key, "x")

    def test_end_of_bibliography_is_not_part_of_last_item(self) -> None:
        entries = parse_bibitems("\\bibitem{a} A. B, ``T1,'' 2001.\n\\bibitem{b} C. D, ``T2,'' 2002.\n\\end{thebibliography}\nAfter")
        self.assertEqual([entry.citation_key for entry in entries], ["a", "b"])
        self.assertEqual(entries[1].title, "T2")


class ExtractTests(unittest.TestCase):
    def test_only_cited_bib_entries_and_bibitems(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            version_dir = Path(tmp) / "2605.00001" / "v1"
            version_dir.mkdir(parents=True)
            (version_dir / "refs.bib").write_text(
                "@article{used, title={Used Paper}, author={Doe, Jane}, year={2020}, journal={J}}\n"
                "@article{unused, title={Unused}, year={2019}}\n"
                "@misc{arx, title={Preprint}, eprint={1706.03762}, archivePrefix={arXiv}}\n",
                encoding="utf-8",
            )
            text = "\\cite{used} \\citep{arx}\\bibliography{refs}"
            entries = extract(version_dir, text)
        self.assertEqual([entry.citation_key for entry in entries], ["used", "arx"])
        self.assertEqual(entries[0].authors, ["Jane Doe"])
        self.assertEqual(entries[0].arxiv_id, "2605-00001")
        self.assertEqual(entries[1].eprint, "1706.03762")

    def test_missing_named_bib_falls_back_to_present_bib(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            version_dir = Path(tmp) / "2605-00002" / "v1"
            version_dir.mkdir(parents=True)
            (version_dir / "other.bib").write_text("@book{k, title={Book}, year=1999}", encoding="utf-8")
            entries = extract(version_dir, "\\cite{k}\\bibliography{references}")
        self.assertEqual([entry.citation_key for entry in entries], ["k"])


class DedupTests(unittest.TestCase):
    def test_statement_example(self) -> None:
        v1 = Entry("vaswani2017", "article", "Attention Is All You Need", ["Ashish Vaswani"], 2017)
        v2 = Entry("vit_transformer", "article", "Attention is all you need", ["Ashish Vaswani"], 2017,
                   doi="10.5555/3295222")
        v3 = Entry("vit_transformer", "article", "Attention is all you need", ["Ashish Vaswani"], None)
        rows, maps = dedup({1: [v1], 2: [v2], 3: [v3]})
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["citation_key"], "vit_transformer")
        self.assertEqual((rows[0]["year"], rows[0]["doi"]), (2017, "10.5555/3295222"))
        self.assertEqual(rows[0]["authors"], ["Ashish Vaswani"])
        self.assertEqual(maps, {1: {"vaswani2017": "vit_transformer"}, 2: {}, 3: {}})

    def test_duplicates_inside_one_version_merge_by_doi(self) -> None:
        a = Entry("a", "article", "Some title here", doi="10.1/x", year=2020)
        b = Entry("b", "article", None, ["X Y"], doi="10.1/x", journal="J")
        rows, maps = dedup({1: [a, b]})
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["title"], rows[0]["journal"], rows[0]["authors"]), ("Some title here", "J", ["X Y"]))
        self.assertEqual(len(maps[1]), 1)

    def test_same_title_different_works_not_merged(self) -> None:
        a = Entry("lecun", "article", "Deep learning", ["Yann LeCun"], 2015)
        b = Entry("goodfellow", "book", "Deep learning", ["Ian Goodfellow"], 2016)
        rows, _ = dedup({1: [a, b]})
        self.assertEqual(len(rows), 2)

    def test_reused_key_for_different_work_gets_unique_key(self) -> None:
        old = Entry("smith2020", "article", "Old paper on graphs", doi="10.1/old")
        new = Entry("smith2020", "article", "New paper on proteins", doi="10.1/new")
        rows, maps = dedup({1: [old], 2: [new]})
        self.assertEqual(sorted(row["citation_key"] for row in rows), ["smith2020", "smith2020_2"])
        self.assertEqual(maps, {1: {"smith2020": "smith2020_2"}, 2: {}})

    def test_arxiv_url_fallback(self) -> None:
        rows, _ = dedup({1: [Entry("k", "misc", "T", eprint="1706.03762")]})
        self.assertEqual(rows[0]["url"], "https://arxiv.org/abs/1706.03762")
        self.assertEqual(
            sorted(rows[0]), sorted(["arxiv_id", "citation_key", "entry_type", "title", "authors", "year", "doi", "journal", "url"])
        )


if __name__ == "__main__":
    unittest.main()
