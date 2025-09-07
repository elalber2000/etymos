import csv
import json
import re
from utils import DATA_DIR
import logging
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


DEL_PATTERN = re.compile(r"(?i)(?<!cognado )\bdel\s+(.+?)(?=\s+del| y | o |\s*[.,;)(])")
PARENS_OR_BRACKETS = re.compile(r"\(.*?\)|\[.*?\]")
NON_LANGS = {"adjetivo", "sustantivo", "verbo", "adverbio", "diminutivo",
             'uso', 'origen', 'topónimo', 'toponimo', 'nombre', 'raíz', 'raiz',
             'préstamo', 'prestamo', "pretérito",
             'alfabeto', 'tono', 'Biscuter', 'neologismo', 'siglo', 'sonido', 
            'vocablo', 'femenino', 'prefijo', 'infinitivo', 'participio',
            'seudónimo', 'título', 'sufijo', 'autónimo', 'término', 'general',
            'plural', 'acusativo', 'común', 'aumentativo', 'augmentativo', 
            'endónimo', 'gerundio', 'articulo', 'masculino', 'gentivio',
            'préterito', 'frecuentativo', 'pronombre', 'distintivo', 'presente',
            'cultismo', 'gerundivo', 'anitguo', 'indicativo', 'desusado' 'fonema',
            'acrónimo', 'científico', 'propio', 'coloquialismo', 'nominativo',
            'genitivo', 'apócope', 'neutro', 'cognado', 'obsoleto', 'artículo',
            'partícipio', 'infintivio', 'singular', 'despectivo', 'dativo',
            'imperativo', 'dialectal', 'infintitivo', 'radical', 'ascendiente',
            'tardío:', 'prefixo', 'anverso', 'épónimo', 'epónimo:'}
SKIP_WORDS = {"que", "cual", "era"}
REMOVE_WORDS = {"tardío", "vulgar", "medieval", "preclásico", "medio", "clásico",
                "medio", "occidental", "bajo", 'imperial', "antiguo", 'nan', 'alto'}
POS_KEEP = {"noun", "verb", "adjective", "adverb"}


def extract_etymology(text: str) -> list[tuple[str, str]]:
    """
    Extracts a list of languages and words from the etymological list

    Args
    - text (str): The etymological text to extract from

    Returns
    (list[tuple[str, str]]): A list of the (word,lang) present in the text
    """
    results = []
    
    for phrase in DEL_PATTERN.findall(text):
        old_lang = ""

        # Remove (...) or [...] content
        phrase = PARENS_OR_BRACKETS.sub("", phrase)

        # Normalize spaces
        phrase = re.sub(r"\s+", " ", phrase).strip()
        words = phrase.split()

        # Skip words that are not langs
        if len(words)>1 and words[1] in SKIP_WORDS:
            continue

        # Remove unwanted words "tardío" and "vulgar"
        if words[0].lower() in REMOVE_WORDS:
            words.pop(0)
        if words[0].lower() in REMOVE_WORDS:
            words.pop(0)
        if len(words)>1 and words[1].lower() in REMOVE_WORDS:
            words.pop(1)

        # Handle "castellano antiguo", "inglés antiguo", etc as one language token
        if len(words)>1 and words[1] == "antiguo":
            lang, word = " ".join(words[:2]), " ".join(words[2:])
        else:
            lang, word = words[0], " ".join(words[1:])

        # No word
        if word == "":
            continue

        # Implicit lang
        if lang in NON_LANGS|REMOVE_WORDS:
            if old_lang=="":
                continue
            else:
                lang=old_lang
        else:
            old_lang = lang

        results.append((lang, word))

    return results


def stream_to_csv(
    file_path: str,
    out_csv: str,
    max_lines: int | None = None,
    batch_size: int = 50_000,
    dedup = True,
):
    """
    Stream through a large JSONL file and save extracted etymology data to CSV.
    Columns: lang_origin,word_origin,lang_dest,word_dest,dump_row
    """
    with open(out_csv, "w", newline="", encoding="utf-8") as csvfile:
        logging.info(f"Opened {out_csv}")
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["lang_origin", "word_origin", "lang_dest", "word_dest", "link_type", "dump_row"]
        )
        writer.writeheader()

        buffer: list[dict[str, str]] = []
        seen = set()

        with open(file_path, "r", encoding="utf-8") as f:
            logging.info(f"Opened {file_path}")
            #for i, line in enumerate(f):
            for i, line in enumerate(tqdm(f, desc="Processing", unit="lines"), start=1):
                if max_lines is not None and i >= max_lines:
                    break
                try:
                    json_line = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if json_line.get("pos") not in POS_KEEP:
                    continue

                ety = json_line.get("etymology_text")
                word_dest = json_line.get("word")
                if not ety or not word_dest:
                    continue

                try:
                    pairs = extract_etymology(f"{ety}.")
                except Exception:
                    continue

                prev_word, prev_lang = None, None

                logging.debug(f"Added words: {pairs}")
                for i_pairs, (lang_origin, word_origin) in enumerate(pairs):
                    if not word_origin:
                        continue

                    key = (lang_origin, word_origin, "castellano", word_dest, "direct" if i_pairs==0 else "indirect")
                    if key not in seen:
                        seen.add(key)
                        buffer.append(
                            {
                                "lang_origin": lang_origin,
                                "word_origin": word_origin,
                                "lang_dest": "castellano",
                                "word_dest": word_dest,
                                "link_type": "direct" if i_pairs==0 else "indirect",
                                "dump_row": i,
                            }
                        )

                        if prev_word is not None and prev_lang is not None:
                            key = (lang_origin, word_origin, "castellano", word_dest, "direct")
                            if key not in seen:
                                seen.add(key)
                                buffer.append(
                                    {
                                        "lang_origin": lang_origin,
                                        "word_origin": word_origin,
                                        "lang_dest": prev_lang,
                                        "word_dest": prev_word,
                                        "link_type": "direct",
                                        "dump_row": i,
                                    }
                                )
                        prev_word, prev_lang = word_origin, lang_origin
                    
                if len(buffer) >= batch_size:
                    writer.writerows(buffer)
                    buffer.clear()

        logging.info("Saving data")
        if buffer:
            writer.writerows(buffer)


if __name__ == "__main__":
    input_path = f"{DATA_DIR}/es-extract.jsonl"
    output_path = f"{DATA_DIR}/dataset.csv"

    stream_to_csv(input_path, output_path)