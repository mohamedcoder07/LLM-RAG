import os
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings(action="ignore")

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.documents import Document

from llama_parse import LlamaParse
from unstructured.partition.pdf import partition_pdf
from io import BytesIO

RAW_MARKDOWN_FILE = "data/markdown/airfrance_brut.md"

def load_docs(folder_path):
    print("[Step 1] DATA COLLECTION")

    docs_path = sorted(Path(folder_path).glob("*.pdf"))

    print(f"[INFO] Nombre de PDFs trouvés : {len(docs_path)}")   


    if Path(RAW_MARKDOWN_FILE).exists():
        loader = TextLoader(RAW_MARKDOWN_FILE, encoding="utf-8")
        documents = loader.load()
        return documents
    
    
    documents = []

    # # Parsing using `PyMuPDFLoader`
    # for path in docs_path :    
    #     loader = PyMuPDFLoader(file_path=path, 
    #                            extract_tables="markdown", 
    #                            mode="page")
    #     doc_loaded = loader.load()    
    #     documents.extend(doc_loaded) 

    # Parsing using `LlamaParse` 
    parser = LlamaParse(
        result_type="markdown",
        num_workers=4,
        ignore_cache=True,
        premium_mode=True,
        verbose=False,
        merge_tables_across_pages_in_markdown=True,
        adaptive_long_table=True,
        compact_markdown_table=True,
        user_prompt="""Tu es un expert en analyse de documents financiers et en redressement de données mal extraites.
            Ta mission principale est d'extraire l'intégralité du contenu de ce document (texte, paragraphes, titres) de manière fidèle et structurée au format Markdown.

            Attention : Le document contient des tableaux financiers. Le texte qui t'est fourni a été extrait par un outil d'OCR/parsing qui détruit souvent la structure tabulaire originale. Tu ne dois PAS te fier à l'alignement brut fourni.

            Lorsque tu identifies des données tabulaires, tu dois impérativement appliquer les règles de correction suivantes avant de générer le tableau Markdown final :

            1. PIÈGE DE LA CELLULE HAUT-GAUCHE : Très souvent, la toute première cellule du tableau (en haut à gauche) est vide dans le document original. Le parseur tente par erreur de la remplir en aspirant le texte de la ligne d'en-dessous, ce qui détruit tout l'en-tête. Tu dois forcer la création d'une cellule d'en-tête vide (ou la nommer logiquement, ex: "Catégorie" ou " ") pour maintenir l'alignement exact des autres colonnes de l'en-tête.
            2. CORRECTION DES EN-TÊTES : Le parser confond souvent la première ligne de données (les labels des lignes, ex: "Flotte principale") et l'en-tête de colonne. Sépare clairement la colonne des catégories (labels) de la ligne des en-têtes de colonnes (périodes, flux).
            3. RÉALIGNEMENT DES VALEURS (LE PLUS IMPORTANT) : À cause de cellules vides dans le document original, le parser a décalé les nombres de manière anarchique (souvent vers la droite, créant de fausses colonnes vides à la fin). Tu dois réassigner chaque nombre à sa colonne logique.
            4. LOGIQUE FINANCIÈRE : Utilise la cohérence mathématique pour replacer les chiffres. Par exemple : [Valeur début de période] + [Augmentations/Livraisons] - [Diminutions/Annulations] = [Valeur fin de période]. Si un chiffre est sous la mauvaise colonne, déplace-le pour que l'équation soit juste.
            5. NETTOYAGE : Supprime les colonnes fantômes (vides) générées par erreur à l'extrémité droite du tableau.

            Produis un bref résumé des informations clés du tableau juste avant de l'afficher. Restituer la version finale, corrigée, recalculée et parfaitement alignée du tableau au format Markdown. Extrais le reste du document normalement."""
    )

    
    for path in docs_path :          
        parsed_pages = parser.load_data(str(path))
        full_markdown_text = ""
        for idx, page in enumerate(parsed_pages):
            full_markdown_text += f"\n\n\n\n"
            full_markdown_text += page.text
        
        with open(RAW_MARKDOWN_FILE, "w", encoding="utf-8") as f:
                f.write(full_markdown_text)      

        documents.extend([doc.to_langchain_format() for doc in parsed_pages])
        
    return documents






# def select_docs():
#     docs = load_docs()
#     # Not implemented yet / In order to select the relevant docs

#     return docs

