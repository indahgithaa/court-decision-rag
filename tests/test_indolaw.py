from src.data_sources.indolaw import is_holdout_candidate, parse_indolaw_xml


XML = """<putusan id="abc" klasifikasi="pidana-khusus"
sub_klasifikasi="narkotika-dan-psikotropika" lembaga_peradilan="pn-contoh"
provinsi="jabar" url="https://example.test/abc">
<kepala_putusan>putusan no 12 pid sus 2018 pn cth demi keadilan berdasarkan ketuhanan</kepala_putusan>
<identitas>nama lengkap contoh tempat lahir bandung</identitas>
<riwayat_penahanan>terdakwa ditahan penyidik</riwayat_penahanan>
<fakta>tanpa hak memiliki narkotika</fakta>
<pertimbangan_hukum>mempertimbangkan pasal 112</pertimbangan_hukum>
<amar_putusan>mengadili menjatuhkan pidana</amar_putusan>
</putusan>"""


def test_parse_and_accept_indolaw_holdout_candidate() -> None:
    metadata = parse_indolaw_xml(XML, source_path="dataset/abc.xml")

    assert metadata["case_year"] == 2018
    assert metadata["case_number_normalized"] == "12 pid sus 2018 pn cth"
    assert is_holdout_candidate(metadata, excluded_courts=set())


def test_rejects_development_court() -> None:
    metadata = parse_indolaw_xml(XML, source_path="dataset/abc.xml")

    assert not is_holdout_candidate(metadata, excluded_courts={"pn-contoh"})
