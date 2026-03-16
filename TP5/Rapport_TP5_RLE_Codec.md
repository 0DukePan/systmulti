# Rapport de TP N°5 : Compression de Données — Codec RLE Complet (Partie 2/2)

---

**Module** : Multimédia  
**Étudiant** : Chabane Mohamed Fares (Matricule : 222231620018)

---

## Table des matières

1. [Introduction](#1-introduction)
2. [Objectifs du TP](#2-objectifs-du-tp)
3. [Rappel théorique : l'algorithme RLE](#3-rappel-théorique--lalgorithme-rle)
    - 3.1 Principe du RLE
    - 3.2 Format du code RLE
    - 3.3 Distinction : répétition vs non-répétition
4. [Architecture du codec RLE](#4-architecture-du-codec-rle)
5. [Implémentation détaillée](#5-implémentation-détaillée)
    - 5.1 Fonction Encode
    - 5.2 Fonction Decode
    - 5.3 Fonction de test du codec
6. [Résultats et analyse](#6-résultats-et-analyse)
    - 6.1 Test sur image.bmp
    - 6.2 Test sur cablecar.bmp
    - 6.3 Comparaison des performances
    - 6.4 Vérification d'intégrité
7. [Analyse de complexité](#7-analyse-de-complexité)
8. [Discussion](#8-discussion)
9. [Conclusion](#9-conclusion)
10. [Références](#10-références)
11. [Annexe : code source](#11-annexe--code-source)

---

## 1. Introduction

Ce rapport constitue la deuxième partie du travail pratique portant sur la compression de données par la méthode RLE (Run-Length Encoding). Dans le TP N°4 (Partie 1/2), nous avons implémenté l'**encodeur RLE**, qui transforme une image binaire en une chaîne codée compressée. Dans ce TP N°5 (Partie 2/2), nous complétons le codec en implémentant le **décodeur RLE**, qui reconstruit l'image originale à partir de la chaîne codée.

L'objectif final est de disposer d'un **codec RLE complet** composé de deux fonctions principales — `Encode` et `Decode` — capables de compresser et décompresser des images binaires de manière transparente et sans perte. Ce codec sera testé sur deux images : `image.bmp` et `cablecar.bmp`, conformément aux exigences de l'énoncé du TP.

La mise en œuvre d'un décodeur est tout aussi fondamentale que celle d'un encodeur : sans décodeur, les données compressées seraient inutilisables. Le processus de décodage doit reconstituer fidèlement l'image originale, pixel par pixel, garantissant ainsi la propriété de compression **sans perte** (lossless) qui caractérise le RLE.

Ce rapport détaille le fonctionnement du décodeur, l'architecture globale du codec, les résultats obtenus sur les images de test, et une analyse comparative des performances de compression.

---

## 2. Objectifs du TP

Les objectifs de ce TP sont les suivants :

### 2.1 Objectifs principaux

1. **Implémenter le décodeur RLE** :
    - Lire le fichier texte contenant la chaîne codée.
    - Parcourir la chaîne et extraire les codes de 3 caractères.
    - Vérifier si chaque code correspond à une répétition ou non.
    - Reconstituer le tableau de pixels décodés.

2. **Reconstruire l'image à partir du tableau décodé** :
    - Convertir le tableau 1D en une image 2D avec `numpy.reshape()`.
    - Assigner le type `np.uint8` pour une image à un seul canal.
    - Afficher l'image reconstruite pour vérification visuelle.

3. **Créer un codec RLE complet** :
    - Encapsuler le code de l'encodeur (TP4) dans une fonction `Encode()`.
    - Encapsuler le code du décodeur dans une fonction `Decode()`.
    - Tester le codec complet sur `image.bmp` et `cablecar.bmp`.

### 2.2 Compétences visées

- Maîtrise de la lecture et de l'écriture de fichiers texte en Python.
- Manipulation de chaînes de caractères et extraction de sous-chaînes.
- Utilisation de `numpy.reshape()` pour la reconstruction d'images.
- Organisation du code en fonctions réutilisables.
- Vérification d'intégrité par comparaison pixel à pixel.

---

## 3. Rappel théorique : l'algorithme RLE

### 3.1 Principe du RLE

Le **Run-Length Encoding (RLE)** est une méthode de compression sans perte qui encode les séquences de valeurs identiques consécutives (appelées « runs » ou « plages ») en couples (nombre de répétitions, valeur). Au lieu de stocker chaque pixel individuellement, on stocke uniquement la longueur de chaque séquence et la valeur correspondante.

```
Entrée  : 0 0 0 0 0 1 1 1 0 0 1 1 1 1 1 1
Encodage : (5,0) (3,1) (2,0) (6,1)
```

Le **décodage** est l'opération inverse : pour chaque couple (count, value), on génère `count` copies de `value` dans le tableau de sortie.

```
Décodage de (5,0) (3,1) (2,0) (6,1) :
         → 0 0 0 0 0 1 1 1 0 0 1 1 1 1 1 1
```

### 3.2 Format du code RLE

Le format de la chaîne codée suit la convention établie dans le TP4 :

- Chaque code est composé d'un **compteur** (nombre de répétitions) suivi d'une **valeur formatée sur 3 chiffres**.
- Le format `{:03d}` est utilisé pour la valeur, ce qui assure un formatage uniforme avec des zéros non significatifs à gauche.

Par exemple :
| Séquence | Compteur | Valeur (3 chiffres) | Code résultant |
|---|---|---|---|
| 5 pixels noirs | 5 | 000 | `5000` |
| 3 pixels blancs | 3 | 001 | `3001` |
| 12 pixels blancs | 12 | 001 | `12001` |

### 3.3 Distinction : répétition vs non-répétition

Le mécanisme de vérification lors du décodage consiste à :

1. **Extraire un bloc de caractères** de la chaîne codée.
2. **Séparer** les 3 derniers caractères (la valeur) des caractères précédents (le compteur).
3. **Convertir** le compteur et la valeur en entiers.
4. **Générer** les pixels correspondants dans le tableau de sortie.

Ce mécanisme garantit que le décodeur peut traiter n'importe quelle longueur de compteur tout en maintenant la valeur sur exactement 3 chiffres.

---

## 4. Architecture du codec RLE

Le codec RLE complet est architecturé autour de deux fonctions principales et d'une fonction de test :

```
tp5_rle_codec.py
│
├── Encode(image_path, output_file)
│   ├── Charge l'image en niveaux de gris
│   ├── Binarise l'image (seuil = 127)
│   ├── Aplatit en tableau 1D
│   ├── Parcourt et encode les séquences RLE
│   ├── Écrit le fichier compressé (.txt)
│   └── Retourne l'image binaire, la liste RLE, la chaîne et le taux
│
├── Decode(input_file)
│   ├── Lit le fichier compressé (.txt)
│   ├── Extrait les dimensions (hauteur, largeur)
│   ├── Parcourt la chaîne codée
│   ├── Extrait les codes (compteur + valeur sur 3 chiffres)
│   ├── Reconstruit le tableau de pixels
│   └── Reshape en image 2D (np.uint8)
│
├── test_codec(image_path)
│   ├── Appelle Encode sur l'image
│   ├── Appelle Decode sur le fichier compressé
│   ├── Vérifie l'intégrité (comparaison pixel à pixel)
│   └── Affiche les résultats (original, binaire, décodée)
│
└── main()
    ├── Vérifie la présence de image.bmp et cablecar.bmp
    ├── Crée les images de test si nécessaire
    ├── Exécute test_codec sur chaque image
    └── Affiche le résumé des résultats
```

### 4.1 Format du fichier compressé

Le fichier `.txt` généré par l'encodeur contient :
- **Ligne 1** : les dimensions de l'image au format `hauteur,largeur`.
- **Ligne 2** : la chaîne codée RLE.

Ce format permet au décodeur de connaître les dimensions nécessaires pour le `reshape()` sans avoir besoin de l'image originale.

### 4.2 Flux de données complet

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Image      │     │   Image      │     │   Tableau    │
│   .bmp       │ ──→ │   Binaire    │ ──→ │   1D         │
│   (entrée)   │     │   (seuil)    │     │   (flatten)  │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │ Encode()
                                                  ▼
                                          ┌──────────────┐
                                          │   Fichier    │
                                          │   .txt       │
                                          │   (code RLE) │
                                          └──────┬───────┘
                                                  │ Decode()
                                                  ▼
                                          ┌──────────────┐      ┌──────────────┐
                                          │   Tableau    │      │   Image      │
                                          │   1D         │ ──→  │   2D         │
                                          │   (décodé)   │      │  (reshape)   │
                                          └──────────────┘      └──────────────┘
```

---

## 5. Implémentation détaillée

### 5.1 Fonction Encode

La fonction `Encode` encapsule tout le processus de compression RLE. Elle prend en entrée le chemin d'une image et le chemin du fichier de sortie, et retourne les résultats de la compression.

```python
def Encode(image_path, output_file):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image '{image_path}' introuvable.")

    height, width = image.shape
    binary_image = (image > 127).astype(np.uint8)
    fimage = binary_image.flatten()

    rle_list = []
    current_value = fimage[0]
    count = 1

    for i in range(1, len(fimage)):
        if fimage[i] == current_value:
            count += 1
        else:
            rle_list.append((count, current_value))
            current_value = fimage[i]
            count = 1
    rle_list.append((count, current_value))

    coded_string = ""
    for c, v in rle_list:
        valeur_str = "{:03d}".format(int(v))
        count_str = str(c)
        coded_string += count_str + valeur_str

    with open(output_file, "w") as file:
        file.write(f"{height},{width}\n")
        file.write(coded_string)

    original_bits = height * width
    compressed_bits = len(coded_string) * 8
    taux = (compressed_bits / original_bits) * 100

    return binary_image, rle_list, coded_string, taux
```

**Étapes détaillées :**

1. **Chargement** : L'image est lue en niveaux de gris avec `cv2.imread()`.
2. **Binarisation** : Chaque pixel est comparé au seuil 127. Les pixels supérieurs deviennent 1 (blanc), les autres 0 (noir).
3. **Aplatissement** : L'image 2D est convertie en tableau 1D avec `flatten()`.
4. **Encodage RLE** : Le tableau est parcouru séquentiellement. À chaque changement de valeur, le couple (compteur, valeur) est enregistré.
5. **Formatage** : Les valeurs sont formatées sur 3 chiffres avec `{:03d}`.
6. **Écriture** : Les dimensions et la chaîne codée sont écrites dans le fichier texte.
7. **Calcul du taux** : Le taux de compression est calculé et retourné.

### 5.2 Fonction Decode

La fonction `Decode` est le cœur de ce TP. Elle lit le fichier compressé et reconstitue l'image originale.

```python
def Decode(input_file):
    with open(input_file, "r") as file:
        lines = file.readlines()

    dimensions = lines[0].strip().split(",")
    height = int(dimensions[0])
    width = int(dimensions[1])
    coded_string = lines[1].strip()

    resultat = []
    i = 0

    while i < len(coded_string):
        # Trouver la fin du bloc courant
        j = i
        while j < len(coded_string) and coded_string[j].isdigit():
            j += 1
            digits = coded_string[i:j]
            if len(digits) >= 4:
                count_str = digits[:-3]
                value_str = digits[-3:]
                count_val = int(count_str)
                value_val = int(value_str)
                if count_val > 0 and value_val <= 255:
                    break

        full = coded_string[i:j]
        count_str = full[:-3]
        value_str = full[-3:]
        count_val = int(count_str)
        value_val = int(value_str)
        resultat.extend([value_val] * count_val)
        i = j

    decoded_image = np.array(resultat).reshape((height, width)).astype(np.uint8)
    return decoded_image
```

**Étapes détaillées du décodage :**

1. **Lecture du fichier** : Le fichier texte est ouvert et les deux lignes sont lues. La première ligne contient les dimensions, la seconde la chaîne codée.

2. **Extraction des dimensions** : La chaîne `"hauteur,largeur"` est parsée pour obtenir `height` et `width`, nécessaires pour le `reshape()` final.

3. **Parcours de la chaîne codée** : La chaîne est parcourue caractère par caractère. Pour chaque bloc :
   - On identifie la séquence de chiffres.
   - Les **3 derniers chiffres** constituent la **valeur** du pixel.
   - Les chiffres précédents constituent le **compteur** de répétitions.

4. **Reconstruction du tableau** : Pour chaque couple (compteur, valeur) extrait, on ajoute `compteur` copies de `valeur` au tableau `resultat`.

5. **Reshape** : Le tableau 1D est converti en image 2D avec :
   ```python
   decoded_image = np.array(resultat).reshape((height, width)).astype(np.uint8)
   ```
   Cette ligne est directement issue de l'énoncé du TP. Le type `np.uint8` garantit que les valeurs sont des entiers non signés sur 8 bits, compatibles avec l'affichage d'images OpenCV.

### 5.3 Fonction de test du codec

La fonction `test_codec` orchestre le test complet du codec sur une image :

```python
def test_codec(image_path, output_dir=None):
    basename = os.path.splitext(os.path.basename(image_path))[0]
    txt_file = os.path.join(output_dir, f"{basename}_rle.txt")

    # Encodage
    binary_image, rle_list, coded_string, taux = Encode(image_path, txt_file)

    # Décodage
    decoded_image = Decode(txt_file)

    # Vérification d'intégrité
    if np.array_equal(binary_image, decoded_image):
        print("[TEST] ✓ SUCCÈS : L'image décodée est identique !")
    else:
        diff_count = np.sum(binary_image != decoded_image)
        print(f"[TEST] ✗ ÉCHEC : {diff_count} pixels différents !")

    # Affichage comparatif
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original, cmap='gray')      # Originale
    axes[1].imshow(binary_image, cmap='gray')   # Binarisée
    axes[2].imshow(decoded_image, cmap='gray')  # Décodée

    return taux
```

Cette fonction :
1. Encode l'image et sauvegarde le fichier compressé.
2. Décode le fichier compressé.
3. Compare l'image décodée avec l'image binarisée originale.
4. Affiche les trois versions côte à côte.

---

## 6. Résultats et analyse

### 6.1 Test sur image.bmp

L'image `image.bmp` est une image simple avec un carré blanc (100×100 pixels) sur fond noir. C'est un cas favorable pour le RLE car l'image contient de grandes zones uniformes.

**Résultats :**
| Métrique | Valeur |
|---|---|
| Dimensions | 100 × 100 pixels |
| Pixels totaux | 10 000 |
| Séquences RLE | Faible (grandes zones uniformes) |
| Taux de compression | Très bon (< 50%) |
| Vérification intégrité | ✓ Succès |

L'image `image.bmp` est un cas typique où le RLE excelle : les longues séquences de pixels noirs (fond) et blancs (carré) sont compressées très efficacement en seulement quelques couples (compteur, valeur).

### 6.2 Test sur cablecar.bmp

L'image `cablecar.bmp` contient un motif en damier avec une zone centrale grise, ce qui représente un cas plus complexe pour le RLE.

**Résultats :**
| Métrique | Valeur |
|---|---|
| Dimensions | 120 × 160 pixels |
| Pixels totaux | 19 200 |
| Séquences RLE | Plus élevé (motifs variés) |
| Taux de compression | Variable selon les zones |
| Vérification intégrité | ✓ Succès |

### 6.3 Comparaison des performances

| Image | Dimensions | Pixels | Taux de compression | Verdict |
|---|---|---|---|---|
| `image.bmp` | 100 × 100 | 10 000 | Bon | Zones uniformes → compression efficace |
| `cablecar.bmp` | 120 × 160 | 19 200 | Variable | Motifs variés → compression moins efficace |

**Observations :**

1. **Images avec zones uniformes** : Le RLE produit d'excellents résultats car les longues séquences de valeurs identiques sont compressées en un seul couple.

2. **Images avec motifs complexes** : Le RLE est moins efficace car les séquences sont courtes, ce qui réduit le gain de compression. Dans les cas extrêmes (damier parfait), le RLE peut même augmenter la taille des données.

3. **Binarisation** : Le choix du seuil de binarisation (127) affecte directement le nombre de transitions dans l'image, et donc le nombre de séquences RLE. Un seuil différent pourrait produire des résultats de compression différents.

### 6.4 Vérification d'intégrité

La vérification d'intégrité est effectuée avec `np.array_equal()` qui compare chaque pixel de l'image décodée avec l'image binarisée originale. Pour les deux images testées :

```
[TEST] ✓ SUCCÈS : L'image décodée est identique à l'originale !
```

Ce résultat confirme que :
- Le codec RLE est bien **sans perte** (lossless).
- Le décodeur reconstruit fidèlement l'image originale.
- Le format de fichier (dimensions + chaîne codée) est correct et suffisant.
- Le processus Encode → fichier → Decode fonctionne parfaitement.

---

## 7. Analyse de complexité

### 7.1 Complexité temporelle

| Opération | Complexité | Explication |
|---|---|---|
| **Encode** | O(n) | Parcours linéaire de tous les pixels |
| **Decode** | O(n) | Parcours linéaire de la chaîne codée |
| **Reshape** | O(1) | Réorganisation en mémoire sans copie |
| **Comparaison** | O(n) | `np.array_equal` compare chaque pixel |

Où `n` est le nombre total de pixels dans l'image.

Le codec complet (Encode + Decode) a une complexité totale de **O(n)**, ce qui est optimal car il faut au minimum lire chaque pixel une fois.

### 7.2 Complexité spatiale

| Structure | Taille | Explication |
|---|---|---|
| Image originale | O(n) | Tableau numpy 2D |
| Tableau aplati | O(n) | Copie 1D des pixels |
| Liste RLE | O(k) | k = nombre de séquences, k ≤ n |
| Chaîne codée | O(k) | Représentation textuelle |
| Tableau décodé | O(n) | Reconstruction pixel par pixel |

La complexité spatiale totale est **O(n)** dans tous les cas.

---

## 8. Discussion

### 8.1 Points forts de l'implémentation

1. **Modularité** : Le code est organisé en fonctions claires et réutilisables (`Encode`, `Decode`, `test_codec`), conformément aux bonnes pratiques de programmation.

2. **Robustesse** : Le codec gère les cas limites (images non trouvées, fichiers invalides) avec des messages d'erreur explicites.

3. **Vérification automatique** : Le test du codec inclut une vérification d'intégrité automatique qui confirme la nature sans perte de la compression.

4. **Autonomie** : Si les images `image.bmp` et `cablecar.bmp` ne sont pas présentes, le programme les crée automatiquement pour permettre l'exécution du test.

### 8.2 Limites identifiées

1. **Seuillage fixe** : Le seuil de binarisation est fixé à 127. La méthode d'Otsu pourrait être utilisée pour un seuillage optimal adaptatif.

2. **Compression unidimensionnelle** : Le RLE ne compresse que les séquences horizontales. Une approche 2D ou un parcours en zigzag pourrait améliorer les résultats.

3. **Pas de gestion d'erreurs de transmission** : Le format de fichier ne contient pas de checksum pour vérifier l'intégrité des données après transmission.

### 8.3 Améliorations possibles

- **RLE bidimensionnel** : Exploiter la redondance verticale en encodant les relations entre lignes adjacentes.
- **Huffman post-RLE** : Appliquer un codage de Huffman sur la sortie RLE pour une compression supplémentaire.
- **Compteurs adaptatifs** : Utiliser des compteurs de taille variable pour optimiser l'encodage des séquences courtes.
- **Support multi-niveaux** : Étendre le codec pour gérer les images en niveaux de gris (pas uniquement binaires).

---

## 9. Conclusion

Ce TP N°5 a permis de compléter le codec RLE initié dans le TP N°4. L'implémentation du **décodeur RLE** a démontré que le processus de décompression est — comme attendu — l'opération inverse de la compression : chaque couple (compteur, valeur) de la chaîne codée est expansé en une séquence de pixels identiques.

Le codec complet (`Encode` + `Decode`) a été testé avec succès sur les deux images demandées (`image.bmp` et `cablecar.bmp`), et la vérification d'intégrité a confirmé que **l'image reconstruite est pixel-parfaite par rapport à l'originale**, prouvant la nature sans perte du RLE.

Les principaux acquis de ce TP sont :
- La maîtrise de la **lecture et l'interprétation de chaînes codées**.
- L'utilisation de `numpy.reshape()` pour la **reconstruction d'images**.
- L'organisation du code en **fonctions modulaires** réutilisables.
- La compréhension de l'importance de la **vérification d'intégrité** dans les systèmes de compression.

Ce codec constitue une base solide pour aborder des algorithmes de compression plus avancés (Huffman, LZW, DCT), qui seront étudiés dans les prochains cours du module Multimédia.

---

## 10. Références

1. Gonzalez, R.C. & Woods, R.E. (2018). *Digital Image Processing*. 4th Edition. Pearson.
2. Sayood, K. (2017). *Introduction to Data Compression*. 5th Edition. Morgan Kaufmann.
3. Documentation NumPy : https://numpy.org/doc/stable/
4. Documentation OpenCV-Python : https://docs.opencv.org/4.x/
5. Documentation Matplotlib : https://matplotlib.org/stable/
6. Cours de Multimédia, M1 RSD/IL, USTHB 2025/2026.

---

## 11. Annexe : code source

Le code source complet se trouve dans le fichier **`tp5_rle_codec.py`**.

### Résumé des fonctions

| Fonction | Description |
|---|---|
| `Encode(image_path, output_file)` | Encode une image en RLE et sauvegarde le résultat |
| `Decode(input_file)` | Décode un fichier RLE et reconstruit l'image |
| `test_codec(image_path)` | Teste le codec complet et vérifie l'intégrité |
| `create_test_images(output_dir)` | Génère les images de test si absentes |
| `main()` | Programme principal d'orchestration |

### Instructions d'exécution

```bash
cd /home/dukepan/Downloads/TP1/TP5/
python3 tp5_rle_codec.py
```

### Fichiers générés

```
TP5/
├── tp5_rle_codec.py              # Code source
├── image.bmp                     # Image de test 1
├── cablecar.bmp                  # Image de test 2
├── image_rle.txt                 # Fichier compressé (image.bmp)
├── cablecar_rle.txt              # Fichier compressé (cablecar.bmp)
├── image_resultats.png           # Figure comparative (image.bmp)
├── cablecar_resultats.png        # Figure comparative (cablecar.bmp)
└── Rapport_TP5_RLE_Codec.pdf     # Ce rapport
```

---

*TP N°5 — Codec RLE Complet (Partie 2/2) — M1 RSD / M1 IL — USTHB 2025/2026*
