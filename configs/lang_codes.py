# -*- coding: utf-8 -*-
# convert the language codes to the language names
# the default <ISO-1 : language> codes
ISO1_lang_codes={
    "af" : "Afrikaans" ,
    "am" : "Amharic" ,
    "ar" : "Arabic" ,
    "hy" : "Armenian" ,
    "as" : "Assamese" ,
    "ast" : "Asturian" ,
    "az" : "Azerbaijani" ,
    "be" : "Belarusian" ,
    "bn" : "Bengali" ,
    "bs" : "Bosnian" ,
    "bg" : "Bulgarian" ,
    "my" : "Burmese" ,
    "ca" : "Catalan" ,
    "ceb" : "Cebuanoa" ,
    "zh" : "Chinese, simplified" ,
    "hr" : "Croatian" ,
    "cs" : "Czech" ,
    "da" : "Danish" ,
    "nl" : "Dutch" ,
    "en" : "English" ,
    "et" : "Estonian" ,
    "tl" : "Tagalog" ,
    "fi" : "Finnish" ,
    "fr" : "French" ,
    "ff" : "Fula" ,
    "gl" : "Galician" ,
    "lg" : "Ganda" ,
    "ka" : "Georgian" ,
    "de" : "German" ,
    "el" : "Greek" ,
    "gu" : "Gujarati" ,
    "ha" : "Hausa" ,
    "he" : "Hebrew" ,
    "hi" : "Hindi" ,
    "hu" : "Hungarian" ,
    "is" : "Icelandic" ,
    "ig" : "Igbo" ,
    "id" : "Indonesian" ,
    "ga" : "Irish" ,
    "it" : "Italian" ,
    "ja" : "Japanese" ,
    "jv" : "Javanese" ,
    "kea" : "Kabuverdianu" ,
    "km" : "Khmer" ,
    "kn" : "Kannada" ,
    "kk" : "Kazakh" ,
    "km" : "Khmer" ,
    "ko" : "Korean" ,
    "ky" : "Kirghiz" ,
    "lo" : "Lao" ,
    "lv" : "Latvian" ,
    "ln" : "Lingala" ,
    "lt" : "Lithuanian" ,
    "luo" : "Luo" ,
    "lb" : "Luxembourgish" ,
    "mk" : "Macedonian" ,
    "ms" : "Malay" ,
    "ml" : "Malayalam" ,
    "mt" : "Maltese" ,
    "mi" : "Maori" ,
    "mr" : "Marathi" ,
    "mn" : "Mongolian" ,
    "ne" : "Nepali" ,
    "ns" : "Northern Sotho" ,
    "nb" : "Norwegian Bokmål" ,
    "ny" : "Nyanja" ,
    "oc" : "Occitan" ,
    "or" : "Oriya" ,
    "om" : "Oromo" ,
    "ps" : "Pashto" ,
    "fa" : "Persian" ,
    "pl" : "Polish" ,
    "pt" : "Portuguese" ,
    "pa" : "Punjabi" ,
    "ro" : "Romanian" ,
    "ru" : "Russian" ,
    "sr" : "Serbian" ,
    "sn" : "Shona" ,
    "sd" : "Sindhi" ,
    "sk" : "Slovak" ,
    "sl" : "Slovenian" ,
    "so" : "Somali" ,
    "ku" : "Kurdish" ,
    "es" : "Spanish" ,
    "sw" : "Swahili" ,
    "sv" : "Swedish" ,
    "tg" : "Tajik" ,
    "ta" : "Tamil" ,
    "te" : "Telugu" ,
    "th" : "Thai" ,
    "tr" : "Turkish" ,
    "uk" : "Ukrainian" ,
    "um" : "Umbundu" ,
    "ur" : "Urdu" ,
    "uz" : "Uzbek" ,
    "vi" : "Vietnamese" ,
    "cy" : "Welsh" ,
    "wo" : "Wolof" ,
    "xh" : "Xhosa" ,
    "yo" : "Yoruba" ,
    "zu" : "Zulu" ,
}
# the extended <ISO-2: language> codes
ISO2_lang_codes ={
    "afr" : "Afrikaans" ,
    "amh" : "Amharic" ,
    "ara" : "Arabic" ,
    "hye" : "Armenian" ,
    "asm" : "Assamese" ,
    "ast" : "Asturian" ,
    "azj" : "Azerbaijani" ,
    "bel" : "Belarusian" ,
    "ben" : "Bengali" ,
    "bos" : "Bosnian" ,
    "bul" : "Bulgarian" ,
    "mya" : "Burmese" ,
    "cat" : "Catalan" ,
    "ceb" : "Cebuanoa" ,
    "zho_simpl" : "Chinese, simplified" ,
    "zho_trad" : "Chinese, traditional" ,
    "hrv" : "Croatian" ,
    "ces" : "Czech" ,
    "dan" : "Danish" ,
    "nld" : "Dutch" ,
    "eng" : "English" ,
    "est" : "Estonian" ,
    "tgl" : "Tagalog" ,
    "fin" : "Finnish" ,
    "fra" : "French" ,
    "ful" : "Fula" ,
    "glg" : "Galician" ,
    "lug" : "Ganda" ,
    "kat" : "Georgian" ,
    "deu" : "German" ,
    "ell" : "Greek" ,
    "guj" : "Gujarati" ,
    "hau" : "Hausa" ,
    "heb" : "Hebrew" ,
    "hin" : "Hindi" ,
    "hun" : "Hungarian" ,
    "isl" : "Icelandic" ,
    "ibo" : "Igbo" ,
    "ind" : "Indonesian" ,
    "gle" : "Irish" ,
    "ita" : "Italian" ,
    "jpn" : "Japanese" ,
    "jav" : "Javanese" ,
    "kea" : "Kabuverdianu" ,
    "kam" : "Khmer" ,
    "kan" : "Kannada" ,
    "kaz" : "Kazakh" ,
    "khm" : "Khmer" ,
    "kor" : "Korean" ,
    "kir" : "Kirghiz" ,
    "lao" : "Lao" ,
    "lav" : "Latvian" ,
    "lin" : "Lingala" ,
    "lit" : "Lithuanian" ,
    "luo" : "Luo" ,
    "ltz" : "Luxembourgish" ,
    "mkd" : "Macedonian" ,
    "msa" : "Malay" ,
    "mal" : "Malayalam" ,
    "mlt" : "Maltese" ,
    "mri" : "Maori" ,
    "mar" : "Marathi" ,
    "mon" : "Mongolian" ,
    "npi" : "Nepali" ,
    "nso" : "Northern Sotho" ,
    "nob" : "Norwegian Bokmål" ,
    "nya" : "Nyanja" ,
    "oci" : "Occitan" ,
    "ory" : "Oriya" ,
    "orm" : "Oromo" ,
    "pus" : "Pashto" ,
    "fas" : "Persian" ,
    "pol" : "Polish" ,
    "por" : "Portuguese" ,
    "pan" : "Punjabi" ,
    "ron" : "Romanian" ,
    "rus" : "Russian" ,
    "srp" : "Serbian" ,
    "sna" : "Shona" ,
    "snd" : "Sindhi" ,
    "slk" : "Slovak" ,
    "slv" : "Slovenian" ,
    "som" : "Somali" ,
    "ckb" : "Kurdish, Central" ,
    "spa" : "Spanish" ,
    "swh" : "Swahili" ,
    "swe" : "Swedish" ,
    "tgk" : "Tajik" ,
    "tam" : "Tamil" ,
    "tel" : "Telugu" ,
    "tha" : "Thai" ,
    "tur" : "Turkish" ,
    "ukr" : "Ukrainian" ,
    "umb" : "Umbundu" ,
    "urd" : "Urdu" ,
    "uzb" : "Uzbek" ,
    "vie" : "Vietnamese" ,
    "cym" : "Welsh" ,
    "wol" : "Wolof" ,
    "xho" : "Xhosa" ,
    "yor" : "Yoruba" ,
    "zul" : "Zulu" ,
}
# the <ISO-2 : ISO-1> codes
ISO2to1_codes = {
    "afr" : "af" ,
    "amh" : "am" ,
    "ara" : "ar" ,
    "hye" : "hy" ,
    "asm" : "as" ,
    "ast" : "ast" ,
    "azj" : "az" ,
    "bel" : "be" ,
    "ben" : "bn" ,
    "bos" : "bs" ,
    "bul" : "bg" ,
    "mya" : "my" ,
    "cat" : "ca" ,
    "ceb" : "ceb" ,
    "zho_simpl" : "zh" ,
    "zho_trad" : "zh" ,
    "hrv" : "hr" ,
    "ces" : "cs" ,
    "dan" : "da" ,
    "nld" : "nl" ,
    "eng" : "en" ,
    "est" : "et" ,
    "tgl" : "tl" ,
    "fin" : "fi" ,
    "fra" : "fr" ,
    "ful" : "ff" ,
    "glg" : "gl" ,
    "lug" : "lg" ,
    "kat" : "ka" ,
    "deu" : "de" ,
    "ell" : "el" ,
    "guj" : "gu" ,
    "hau" : "ha" ,
    "heb" : "he" ,
    "hin" : "hi" ,
    "hun" : "hu" ,
    "isl" : "is" ,
    "ibo" : "ig" ,
    "ind" : "id" ,
    "gle" : "ga" ,
    "ita" : "it" ,
    "jpn" : "ja" ,
    "jav" : "jv" ,
    "kea" : "kea" ,
    "kam" : "km" ,
    "kan" : "kn" ,
    "kaz" : "kk" ,
    "khm" : "km" ,
    "kor" : "ko" ,
    "kir" : "ky" ,
    "lao" : "lo" ,
    "lav" : "lv" ,
    "lin" : "ln" ,
    "lit" : "lt" ,
    "luo" : "luo" ,
    "ltz" : "lb" ,
    "mkd" : "mk" ,
    "msa" : "ms" ,
    "mal" : "ml" ,
    "mlt" : "mt" ,
    "mri" : "mi" ,
    "mar" : "mr" ,
    "mon" : "mn" ,
    "npi" : "ne" ,
    "nso" : "ns" ,
    "nob" : "nb" ,
    "nya" : "ny" ,
    "oci" : "oc" ,
    "ory" : "or" ,
    "orm" : "om" ,
    "pus" : "ps" ,
    "fas" : "fa" ,
    "pol" : "pl" ,
    "por" : "pt" ,
    "pan" : "pa" ,
    "ron" : "ro" ,
    "rus" : "ru" ,
    "srp" : "sr" ,
    "sna" : "sn" ,
    "snd" : "sd" ,
    "slk" : "sk" ,
    "slv" : "sl" ,
    "som" : "so" ,
    "ckb" : "ku" ,
    "spa" : "es" ,
    "swh" : "sw" ,
    "swe" : "sv" ,
    "tgk" : "tg" ,
    "tam" : "ta" ,
    "tel" : "te" ,
    "tha" : "th" ,
    "tur" : "tr" ,
    "ukr" : "uk" ,
    "umb" : "um" ,
    "urd" : "ur" ,
    "uzb" : "uz" ,
    "vie" : "vi" ,
    "cym" : "cy" ,
    "wol" : "wo" ,
    "xho" : "xh" ,
    "yor" : "yo" ,
    "zul" : "zu" ,
}
# the <ISO-2_lang family: ISO-2 >
ISO2wFamily_ISO2codes = {
    "afr_Latn": "afr",
    "amh_Ethi": "amh",
    "arb_Arab": "ara",
    "asm_Beng": "asm",
    "ast_Latn": "ast",
    "azj_Latn": "azj",
    "bel_Cyrl": "bel",
    "ben_Beng": "ben",
    "bos_Latn": "bos",
    "bul_Cyrl": "bul",
    "cat_Latn": "cat",
    "ceb_Latn": "ceb",
    "ces_Latn": "ces",
    "ckb_Arab": "ckb",
    "cym_Latn": "cym",
    "dan_Latn": "dan",
    "deu_Latn": "deu",
    "ell_Grek": "ell",
    "eng_Latn": "eng",
    "est_Latn": "est",
    "fin_Latn": "fin",
    "fra_Latn": "fra",
    "fuv_Latn": "ful",
    "gle_Latn": "gle",
    "glg_Latn": "glg",
    "guj_Gujr": "guj",
    "hau_Latn": "hau",
    "heb_Hebr": "heb",
    "hin_Deva": "hin",
    "hrv_Latn": "hrv",
    "hun_Latn": "hun",
    "hye_Armn": "hye",
    "ibo_Latn": "ibo",
    "ind_Latn": "ind",
    "isl_Latn": "isl",
    "ita_Latn": "ita",
    "jav_Latn": "jav",
    "jpn_Jpan": "jpn",
    "kam_Latn": "kam",
    "kan_Knda": "kan",
    "kat_Geor": "kat",
    "kaz_Cyrl": "kaz",
    "khm_Khmr": "khm",
    "kir_Cyrl": "kir",
    "kor_Hang": "kor",
    "lao_Laoo": "lao",
    "lij_Latn": "lav",
    "lim_Latn": "kea",
    "lin_Latn": "lin",
    "lit_Latn": "lit",
    "ltz_Latn": "ltz",
    "lug_Latn": "lug",
    "luo_Latn": "luo",
    "lvs_Latn": "lav",
    "mal_Mlym": "mal",
    "mar_Deva": "mar",
    "mkd_Cyrl": "mkd",
    "mlt_Latn": "mlt",
    "khk_Cyrl": "mon",
    "mri_Latn": "mri",
    "mya_Mymr": "mya",
    "nld_Latn": "nld",
    "nob_Latn": "nob",
    "npi_Deva": "npi",
    "nso_Latn": "nso",
    "nya_Latn": "nya",
    "oci_Latn": "oci",
    "gaz_Latn": "orm",
    "ory_Orya": "ory",
    "pan_Guru": "pan",
    "pes_Arab": "fas",
    "pol_Latn": "pol",
    "por_Latn": "por",
    "pbt_Arab": "pus",
    "ron_Latn": "ron",
    "rus_Cyrl": "rus",
    "slk_Latn": "slk",
    "sna_Latn": "sna",
    "snd_Arab": "snd",
    "som_Latn": "som",
    "spa_Latn": "spa",
    "srp_Cyrl": "srp",
    "swe_Latn": "swe",
    "swh_Latn": "swh",
    "tam_Taml": "tam",
    "tel_Telu": "tel",
    "tgk_Cyrl": "tgk",
    "tgl_Latn": "tgl",
    "tha_Thai": "tha",
    "tur_Latn": "tur",
    "ukr_Cyrl": "ukr",
    "umb_Latn": "umb",
    "urd_Arab": "urd",
    "uzn_Latn": "uzb",
    "vie_Latn": "vie",
    "wol_Latn": "wol",
    "xho_Latn": "xho",
    "yor_Latn": "yor",
    "zho_Hans": "zho_simpl",
    "zho_Hant": "zho_trad",
    "zsm_Latn": "msa",
    "zul_Latn": "zul",
}

class LangCodes:
    def __init__(self,):
        self.ISO1_lang_codes = ISO1_lang_codes
        self.ISO2_lang_codes = ISO2_lang_codes
        self.ISO2wFamily_ISO2codes = ISO2wFamily_ISO2codes
        self.ISO2to1_codes = ISO2to1_codes

    def get_lang(self, lang_code:str)->str:
        """
        traverse all lang_codes, return the language name for a lang_codes
        """
        if lang_code in self.ISO1_lang_codes:
            return self.ISO1_lang_codes[lang_code]
        if lang_code in self.ISO2_lang_codes:
            return self.ISO2_lang_codes[lang_code]
        if lang_code in self.ISO2wFamily_ISO2codes:
            iso2_code = self.ISO2wFamily_ISO2codes[lang_code]
            return self.ISO2_lang_codes[iso2_code]
        else:
            return "unknown language"
    def get_family(self, lang_code:str)->str:
        """
        only support ISO2code with family, return the language family
        """
        if lang_code in self.ISO2wFamily_ISO2codes:
            return lang_code.split("_")[1]
        else:
            return "unknown language family"

    def check_support(self, lang_code:str)->bool:
        """
        check if the language is supported
        """
        return (lang_code in self.ISO1_lang_codes) or \
                (lang_code in self.ISO2_lang_codes) or \
                (lang_code in self.ISO2wFamily_ISO2codes)
