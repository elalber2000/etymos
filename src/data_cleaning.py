from utils import DATA_DIR
import pandas as pd

non_langs = {
    'calendario', 'abedul', 'protosemita', 'algún', 'mismo', 
    'primer', 'anterior', 'cual', 'mapuche', 'pez', 
    'monte', 'magrebí', 'DRAE', 'cristianismo', 'segmento', 
    'Oriente', 'que', 'vocabulario', 'pidgin', 'etimo', 
    'cuento', 'poeta', 'mātrīx', 'arcaico', 'poema', 
    'líquido', 'grafito', 'del', 'personaje', 'tipo', 
    'planeta', 'dios', 'hatiano', 'دار', 'pampa', 
    'romano', 'ser', 'habla', 'físico', 'sistema', 
    'animal', 'comercio', 'Diario', 'Sol', 'sol', 
    'máximo', 'apellido', 'susodicho', 'jefe', 'insecto', 
    'tricornio', 'botánico', 'caput', 'pasado', 'reloj', 
    'claśico', 'mariscal', 'grito', 'municipio', 'guugu', 
    'franco', 'vándalo', 'dialecto', 'Imperio', 'cuerpo', 
    'idioma', '+', 'derecho', 'escandinado', 'rey', 
    'balbuceo', 'estado', '28', 'primigenio', 'diccionario', 
    'judaísmo', 'spiced', 'British', 'locativo', 'médico', 
    'desusado', 'verso', 'ruido', 'griegoτραχεῖα', 'inframundo', 
    'norte', 'fonema', 'joisano', 'químico', 'Lenguaje', 
    'canto', 'mineral', 'panda', 'vapor', 'inglésTELetype', 
    'Himno', 'género', 'Barón', 'modelo', 'cantar', 
    'continente', 'todo', 'antepasado', 'alemánLyserg', 'dedo', 
    'atolón', 'presidente', 'Glasgow', 'dicho', 'fil:', 
    'año', 'Libro', 'Templo', 'pecho', 'lenguaje', 
    'sacerdote', 'criollo', 'río', 'tema', 'teólogo', 
    'tabaco', 'sitio', 'teclado', 'generalFrancisco', 'tūber', 
    'Estado', 'segundo', 'Oráculo', 'batir', 'movimiento', 
    'himno', 'apodo', 'singulus', 'maturare', 'ave', 
    'equivalente', 'fruto', 'sede', 'Talmud', 'cacique', 
    'lugar', 'octavo', 'numeral', 'error', 'más', 
    'privativo', 'temprano', 'noreste', 'paraguano', 'nombramiento', 
    'celtolatino', 'oxígeno', 'viejo', 'sentido', 'latínsubtilitas', 
    'chasquido', 'Reino', 'navío', 'tripulante', 'Palacio', 
    'pabellón', 'tintineo', 'fotógrafo', 'filme', 'matemático', 
    'postclásico', 'praclásico', 'sinónimo', 'mesodermo', 'grupo', 
    'agua', 'templo', 'epónimo', 'germano-latino', 'antedicho', 
    'substantivo', 'Ingachunchullu', 'modo', 'celto-latino', 'matemáticonorteamericano', 
    'Matemático', 'Infierno', 'pelvi', 'método', 'filósofo', 
    'folclore', 'persona', 'cruce', 'parpar', 'planetary-mass', 
    'periodo', 'significado', 'elemento', 'perfecto', 'Perú', 
    'Cruz', 'Fútbol', 'latínblatta', 'baile', 'fráncico–', 
    'caso', 'primero', 'tercer', 'latínmǎcǔla:', '-o', 
    'curtāre', 'tallo', 'miércoles', 'minio', 'latínglattīre', 
    'autor', 'Gobierno', 'abecedario', 'sur', 'piel', 
    'intensificador', 'senador', 'Marco', 'calamus', 'furioso', 
    'desierto', 'imperio', 'Marruecos', 'músico', 'fisiólogo', 
    'servicio', 'político', 'día', 'explorador', 'juego', 
    'proceso', 'libro', 'frase', 'amigo', 'austríaco', 
    'inventor', 'חמש', 'período', 'American', 'Génesis', 
    'toki', 'RIKEN', 'ICIN', 'distrito', 'metal', 
    'hocico', 'latínlacusta', 'Invierno', 'caballo', 'circo', 
    'inmigrante', 'Monte', 'reino', 'célebre', 'laboratorio', 
    'príncipe', 'Salón', 'pasillo', 'árbol', 'patriarca', 
    'derivado', 'proto', 'proto-indoeuropeo*meg', "periodismo",
    "dispositivo",
}

lang_vars = {
    "español": ["español", "castellano", "español:", "andaluz", "leonés"],
    "gallego-portugués": ["galaicoportugués", "galacioportugués", "portugués", "gallego", "mirandés"],
    "francés": ["francés", "frances", "fancés", "francico", "fráncico", "fŕancico", "franc?s", "fráncico–", "judeofrancés", "franco", "francoprovenzal", "francoprovenzal", "picardo"],
    "latín": ["latín", "latin", "Latín", "Latino", "neolatín", "latino", "latín:", "látin", "protoitálcio"],
    "protoindoeuropeo": ["protoindoeuropeo", "proto-indoeuropeo", "proto-indo-europeo", "Proto-Indoeuropeo", "protoidoeuropeo", "protoindoeropeo", "protoindeuropeo", "protoindoeurpeo", "protoindodeurpeo", "protondoeuropeo"],
    "germánico": ["germánico", "proto-germánico", "protogermánico", "proto‐germánico", "potogermánico", "protogermáncio", "kermánico", "alemán", "alemánico", "alemánLyserg"],
    "inglés": ["inglés", "ingles", "Inglés", "anglosajón", "anglonormando", "anglo-normando", "anglorromaní"],
    "griego": ["griego", "Griego", "greigo", "giego", "griengo", "griego:", "griegoτραχεῖα", "protohelénico", "Proto-Helénico", "micénico"],
    "celta": ["celta", "céltico", "protocelta", "protocéltico", "proto-celta", "protoceltico", "celtíbero", "goidélico", "hispano-céltico", "celtolatino", "celto-latino"],
    "árabe": ["árabe", "arábico", "árabeandalusí", "magrebí", "amazig", "tamazight"],
    "vasco": ["vasco", "vascuence", "euskera", "protovasco"],
    "inupiaq": ["iñupiaq"],
    "alemán": ["alemán:", "áleman"],
    "italiano": ["italiano", "napolitano", "siciliano", "veneciano", "florentino", "milanés", "corso", "lombardo", "véneto", "piamontés", "genovés", "toscano"],
    "escandinavo": ["nórdico", "nordico", "noruego", "sueco", "danés", "islandés", "protonórdico"],
    "eslavo": ["eslavo", "protoeslavo", "eslavónico", "slavo", "búlgaro", "ruso", "polaco", "checo", "ucraniano", "serbio", "serbocroata", "esloveno", "croata", "slavo", "bielorruso"],
    "semítico": ["protosemítico", "protosemitico", "protosemita", "acadio", "asirío", "siríaco", "arameo", "hebreo", "Hebreo", "fenicio", "tigriña", "amárico", "Asirio"],
    "indio-iranio": ["protoindoiranio", "protoindoario", "sánscrito", "avéstico", "prácrito", "persa", "pastún", "hindi", "maratí", "guyaratí", "urdu", "bengalí", "protoiranio", "sogdiano", "sindhi"],
    "chino": ["chino", "mandarín", "cantonés", "wu", "min", "sino-coreano"],
    "japonés": ["japonés", "protojapónico", "okinawense"],
    "coreano": ["coreano", "sino-coreano"],
    "maya": ["maya", "yucateco", "quiché", "huasteco", "pipil", "tojolabal", "triqui", "chol", "poqomam"],
    "quechua": ["quechua", "runasimi"],
    "guaraní": ["guaraní", "ñeengatú", "tupinambá", "mawé"],
    "nahua": ["náhuatl", "nahuatl", "protonahua", "protoyutoazteca", "pipil"],
    "proto-tai": ["Proto-Tai", "proto-Tai", "Tai"],
    "protosinotibetano": ["birmano"],
    "proto-mon-jemer": ["Proto-Mon-Khmer", "protoaustroasiático", "protokátuico"],
    "protoaustronesio": ["protofilipino"],
    "protopolinesio": ["samoano", "tuamotuano"],
    "javanés": ["kawi"],
    "protocaucásico": ["protoabjasio", "abjasio", "proto-avar-andi"],
    "protourálico": ["protomordvínico"],
    "túrquico": ["chagatay", "kirguís", "yakuto"],
    "mongol": ["calmuco"],
    "algonquino": ["miami", "cree"],
    "protoyutoazteca": ["tarahumara"],
    "arawak": ["guajiro", "wayuu", "iñeri"],
    "caribe": ["Caribe", "carib", "protocaribeño", "tamanaco"],
    "proto-chon": ["ona", "haush", "tehuelche"],
    "protobantú": ["luba-kasai", "umbundu", "kimbundu", "quimbundo"],
    "proto-otomí": ["mazahua"],
    "suajili": ["swahili"],
}

if __name__ == "__main__":
    path = f"{DATA_DIR}/dataset.csv"

    df = pd.read_csv(path)
    df = df[~df["lang_origin"].isin(non_langs)]

    variant_to_std = {}
    for std, variants in lang_vars.items():
        for v in variants:
            variant_to_std[v.lower()] = std

    df["lang_origin"] = (
        df["lang_origin"]
        .str.lower()
        .map(variant_to_std)
        .fillna(df["lang_origin"])
    )

    counts = df["lang_origin"].value_counts()
    valid_langs = counts[counts >= 15].index
    df = df[df["lang_origin"].isin(valid_langs)]

    pd.set_option('display.max_rows', 500)
    print(df["lang_origin"].value_counts())