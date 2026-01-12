# Compression de LLM pour l'Informatique de Bord - Guide Complet du Projet

Un projet complet qui compresse les Grands Modèles de Langage (LLM) pour les rendre adaptés aux appareils de bord, avec une interface chatbot interactive pour comparer les performances des modèles.

---

## 📖 Ce Que Ce Projet Fait

Ce projet démontre comment :
1. **Compresser de grands modèles IA** (BERT) en versions plus petites et plus rapides (DistilBERT)
2. **Réduire la taille des modèles** jusqu'à 85% tout en maintenant une bonne précision
3. **Comparer les modèles** côte à côte en utilisant une interface web
4. **Préparer les modèles** pour le déploiement sur des appareils à ressources limitées (téléphones, appareils IoT, etc.)

---

## 🎯 Vue d'Ensemble du Projet

### Partie 1 : Pipeline de Compression de Modèle (Notebook Jupyter)
Le notebook (`Extreme_LLM_Compression_for_Edge_Computing.ipynb`) contient un pipeline complet qui :

- **Entraîne** un grand modèle BERT (109M paramètres) sur l'analyse de sentiment
- **Le compresse** en utilisant plusieurs techniques :
  - Distillation de connaissances (transfert de connaissances vers un modèle plus petit)
  - Élagage (suppression de poids inutiles)
  - Quantification (réduction de précision de 32 bits à 8 bits)
  - Conversion ONNX (optimisation pour le déploiement)
- **Mesure** les performances à chaque étape

### Partie 2 : Interface Chatbot Interactive (Application Flask)
Une application web qui vous permet de :
- Saisir du texte et voir les prédictions de sentiment
- Comparer BERT vs modèle compressé côte à côte
- Voir les différences de vitesse et de précision en temps réel

---

## 📁 Structure du Projet

```
dataminingproject/
│
├── Extreme_LLM_Compression_for_Edge_Computing.ipynb  # Pipeline de compression principal
│
├── app.py                    # Application web Flask
├── requirements.txt          # Dépendances Python
│
├── templates/
│   └── index.html           # Interface web
│
├── static/
│   ├── css/
│   │   └── style.css        # Styles
│   └── js/
│       └── script.js        # Fonctionnalités interactives
│
└── models/                   # Modèles sauvegardés (créés après l'entraînement)
    ├── teacher/             # Fichiers du modèle BERT
    └── student/             # Fichiers du modèle compressé
```

---

## 🚀 Guide de Démarrage Rapide

### Étape 1 : Comprendre Ce Dont Vous Avez Besoin

**Requis :**
- Python 3.8 ou supérieur
- Jupyter Notebook (pour exécuter le pipeline de compression)
- Un ordinateur avec GPU (recommandé) ou CPU
- Connexion Internet (pour télécharger les modèles et les jeux de données)

### Étape 2 : Configurer l'Environnement

1. **Ouvrir le Terminal** et naviguer vers le dossier du projet :
   ```bash
   cd dataminingproject
   ```

2. **Créer un environnement virtuel** (optionnel mais recommandé) :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Windows: venv\Scripts\activate
   ```

3. **Installer les packages requis** :
   ```bash
   pip install -r requirements.txt
   ```

   Si vous n'avez pas `requirements.txt`, installez ces packages :
   ```bash
   pip install jupyter transformers datasets torch flask
   ```

### Étape 3 : Exécuter le Pipeline de Compression (Optionnel)

**Cette étape entraîne et compresse les modèles. Vous pouvez l'ignorer si vous voulez simplement utiliser des modèles pré-entraînés.**

1. **Ouvrir Jupyter Notebook** :
   ```bash
   jupyter notebook
   ```

2. **Ouvrir** `Extreme_LLM_Compression_for_Edge_Computing.ipynb`

3. **Exécuter toutes les cellules** séquentiellement (cela prend du temps - 30-60 minutes) :
   - Le notebook va :
     - Télécharger le jeu de données SST-2 de sentiment
     - Entraîner BERT sur l'analyse de sentiment
     - Le compresser en DistilBERT
     - Appliquer l'élagage et la quantification
     - Sauvegarder les modèles dans le dossier `models/`

4. **Sauvegarder les modèles** (ils seront sauvegardés automatiquement dans le notebook)

### Étape 4 : Exécuter l'Interface Chatbot

1. **Assurez-vous d'être dans le répertoire du projet**

2. **Démarrer le serveur Flask** :
   ```bash
   python app.py
   ```

3. **Ouvrir votre navigateur web** et aller à :
   ```
   http://localhost:5001
   ```

4. **Commencer à comparer les modèles !**
   - Tapez n'importe quelle phrase dans la zone de saisie
   - Cliquez sur "Analyser"
   - Voyez les prédictions des deux modèles côte à côte

---

## 📝 Instructions Détaillées Étape par Étape

### Comprendre le Pipeline de Compression

Le notebook suit ces étapes :

#### Étape 1 : Préparation des Données
- Utilise le jeu de données **SST-2** (Stanford Sentiment Treebank)
- Contient des critiques de films étiquetées comme positives ou négatives
- Divise les données en : entraînement (5 000 échantillons), validation (500), test (1 000)

#### Étape 2 : Entraînement du Modèle Enseignant
- **Modèle** : BERT-base-uncased (109 millions de paramètres)
- **Tâche** : Classification binaire de sentiment
- **Entraînement** : 3 époques
- **Résultat** : Modèle qui peut classer le texte comme positif ou négatif

#### Étape 3 : Distillation de Connaissances
- **Modèle Étudiant** : DistilBERT (67 millions de paramètres - 39% plus petit)
- **Processus** : Le modèle plus petit apprend des prédictions du modèle plus grand
- **Résultat** : Modèle compressé avec une précision similaire (88,8% vs 89,4%)

#### Étape 4 : Élagage
- **Méthode** : Supprime 30% des poids les moins importants
- **Résultat** : Maintient la précision tout en réduisant la complexité computationnelle

#### Étape 5 : Quantification
- **Processus** : Convertit les poids de 32 bits à 8 bits de précision
- **Résultat** : 68% de réduction de taille (255MB → 132MB)

#### Étape 6 : Conversion ONNX
- **Format** : Convertit en ONNX (Open Neural Network Exchange)
- **Objectif** : Optimisé pour le déploiement sur divers appareils

#### Étape 7 : Benchmarking
- Mesure la précision, la latence et la taille du modèle à chaque étape
- Crée des graphiques de comparaison et des statistiques

### Comprendre l'Interface Chatbot

L'application Flask fournit :

1. **Interface Interactive**
   - Design web moderne et épuré
   - Saisie de texte facile à utiliser
   - Prédictions en temps réel

2. **Comparaison Côte à Côte**
   - Montre les prédictions des deux modèles
   - Affiche les scores de confiance
   - Montre le temps d'inférence (la vitesse de chaque modèle)

3. **Tableau de Bord de Statistiques**
   - Suit combien de fois les modèles sont d'accord
   - Montre le facteur d'accélération (à quel point le modèle compressé est plus rapide)
   - Affiche le nombre total de requêtes analysées

---

## 🎓 Ce Que Fait Chaque Composant

### Notebook Jupyter (`Extreme_LLM_Compression_for_Edge_Computing.ipynb`)

**Objectif** : Entraîner et compresser les modèles

**Ce qu'il fait** :
- Télécharge et prépare les données
- Entraîne un grand modèle BERT
- Le compresse en utilisant plusieurs techniques
- Mesure les métriques de performance
- Sauvegarde les modèles compressés

**Temps requis** : 30-60 minutes (selon le matériel)

**Sortie** : Modèles entraînés sauvegardés dans le dossier `models/`

### Application Flask (`app.py`)

**Objectif** : Comparer les modèles de manière interactive

**Ce qu'elle fait** :
- Charge les modèles BERT et compressé
- Fournit une interface web pour les tests
- Montre des comparaisons en temps réel
- Calcule les métriques de performance

**Temps de démarrage** : 1-2 minutes (pour charger les modèles)

**Sortie** : Interface web à http://localhost:5001

---

## 📊 Résultats Attendus

Après avoir exécuté le pipeline de compression, vous devriez voir :

| Modèle | Taille | Précision | Vitesse | Paramètres |
|--------|--------|-----------|---------|------------|
| **BERT-base** (Enseignant) | 418 MB | 89,4% | 23,5 ms | 109M |
| **DistilBERT** (Compressé) | 255 MB | 88,8% | 8,2 ms | 67M |
| **Accélération** | **38% plus petit** | **-0,6%** | **2,9x plus rapide** | **39% de moins** |

**Réalisations Clés** :
- ✅ 38% de réduction de taille
- ✅ 2,9x inférence plus rapide
- ✅ Seulement 0,6% de perte de précision
- ✅ 84% de réduction de taille avec quantification (132 MB)

---

## 💡 Comment Utiliser le Chatbot

1. **Démarrer l'application** :
   ```bash
   python app.py
   ```

2. **Attendre que les modèles se chargent** (vous verrez des messages dans le terminal)

3. **Ouvrir le navigateur** : Aller à `http://localhost:5001`

4. **Essayer des exemples** :
   - **Positif** : "Ce film est absolument fantastique !"
   - **Négatif** : "J'ai détesté ce film, il était terrible."
   - **Neutre** : "L'intrigue était correcte, rien de spécial."

5. **Observer les différences** :
   - Vérifier si les deux modèles sont d'accord
   - Comparer les scores de confiance
   - Voir la différence de vitesse (latence)
   - Regarder les statistiques se mettre à jour

---

## 🔧 Dépannage

### Problème : "Le port 5000 est déjà utilisé"
**Solution** : L'application utilise le port 5001 par défaut. Si vous avez besoin d'un autre port :
```bash
FLASK_PORT=8080 python app.py
```

### Problème : Les modèles prennent trop de temps à charger
**Solution** : C'est normal lors de la première exécution. Les modèles téléchargent depuis HuggingFace (~700MB). Les exécutions suivantes seront plus rapides.

### Problème : Erreur "Modèle non trouvé"
**Solution** : 
- Si vous n'avez pas entraîné de modèles, l'application utilisera des modèles pré-entraînés depuis HuggingFace (c'est bien !)
- Pour utiliser vos modèles entraînés, copiez-les dans `models/teacher/` et `models/student/`

### Problème : Erreurs de mémoire insuffisante
**Solution** : 
- Fermez les autres applications
- L'application fonctionne sur CPU si vous n'avez pas de GPU
- Réduisez la taille des lots dans le notebook si vous entraînez

### Problème : Le notebook Jupyter ne démarre pas
**Solution** :
```bash
pip install jupyter
jupyter notebook
```

---

## 🎯 Cas d'Usage

Ce projet est utile pour :

1. **Apprendre** : Comprendre comment fonctionne la compression de modèles
2. **Recherche** : Comparer les techniques de compression
3. **Déploiement** : Préparer les modèles pour les appareils de bord
4. **Éducation** : Enseigner les concepts de compression ML
5. **Développement** : Construire des applications IA légères

---

## 📚 Concepts Clés Expliqués Simplement

### Distillation de Connaissances
**Quoi** : Un modèle "étudiant" plus petit apprend d'un modèle "enseignant" plus grand
**Pourquoi** : Obtenir des performances similaires avec moins de ressources
**Analogies** : Comme un étudiant apprenant d'un enseignant expérimenté

### Élagage
**Quoi** : Supprimer les parties inutiles du modèle
**Pourquoi** : Réduire la taille sans perdre beaucoup de précision
**Analogies** : Comme tailler un arbre - enlever des branches mais le garder en bonne santé

### Quantification
**Quoi** : Utiliser moins de bits pour stocker les nombres (32 bits → 8 bits)
**Pourquoi** : Réduire drastiquement la taille du modèle
**Analogies** : Comme compresser une photo - fichier plus petit, qualité légèrement inférieure

### ONNX
**Quoi** : Format standard pour les modèles IA
**Pourquoi** : Fonctionne sur de nombreux appareils différents
**Analogies** : Comme un format de fichier universel (comme PDF)

---

## 🔄 Résumé du Flux de Travail

```
1. Exécuter le Notebook
   ↓
2. Entraîner le Modèle BERT (Enseignant)
   ↓
3. Compresser en DistilBERT (Étudiant)
   ↓
4. Appliquer l'Élagage et la Quantification
   ↓
5. Sauvegarder les Modèles
   ↓
6. Exécuter l'Application Flask
   ↓
7. Comparer les Modèles dans le Navigateur
   ↓
8. Analyser les Résultats
```

---

## 📦 Prérequis

### Pour Exécuter le Notebook :
- Python 3.8+
- Jupyter Notebook
- GPU recommandé (CPU fonctionne mais plus lent)
- 10GB+ d'espace disque (pour les modèles et les données)

### Pour Exécuter l'Application Flask :
- Python 3.8+
- Flask
- Bibliothèque Transformers
- 2GB+ de RAM
- Navigateur web

### Packages Python :
```
torch
transformers
datasets
flask
numpy
pandas
matplotlib
seaborn
jupyter
```

---

## 🎓 Objectifs d'Apprentissage

Après avoir complété ce projet, vous comprendrez :

1. ✅ Comment entraîner des modèles de transformateurs pour des tâches NLP
2. ✅ La technique de distillation de connaissances
3. ✅ Les méthodes d'élagage de modèles
4. ✅ Les techniques de quantification
5. ✅ Les considérations de déploiement de modèles
6. ✅ Les compromis de performance dans la compression de modèles
7. ✅ La construction d'applications ML interactives

---

## 📖 Ressources Additionnelles

- **Article BERT** : [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
- **Article DistilBERT** : [arXiv:1910.01108](https://arxiv.org/abs/1910.01108)
- **HuggingFace** : https://huggingface.co/
- **ONNX** : https://onnx.ai/

---

## 🤝 Obtenir de l'Aide

Si vous rencontrez des problèmes :

1. Vérifiez attentivement les messages d'erreur
2. Consultez la section de dépannage ci-dessus
3. Assurez-vous que toutes les dépendances sont installées
4. Vérifiez que la version Python est 3.8 ou supérieure
5. Assurez-vous d'avoir assez d'espace disque

---

## ✅ Liste de Vérification Rapide

Avant de commencer :
- [ ] Python 3.8+ installé
- [ ] Dossier du projet téléchargé
- [ ] Accès Terminal/ligne de commande
- [ ] Connexion Internet (pour les téléchargements)

Pour exécuter le notebook :
- [ ] Jupyter installé
- [ ] Tous les packages Python installés
- [ ] GPU disponible (optionnel mais recommandé)

Pour exécuter l'application Flask :
- [ ] Flask installé
- [ ] Modèles disponibles (ou utiliser le repli HuggingFace)
- [ ] Port 5001 disponible
- [ ] Navigateur web installé

---

## 🎉 Indicateurs de Succès

Vous avez réussi à compléter le projet quand :

1. ✅ Le notebook s'exécute sans erreurs
2. ✅ Les modèles sont entraînés et sauvegardés
3. ✅ L'application Flask démarre avec succès
4. ✅ L'interface web se charge dans le navigateur
5. ✅ Vous pouvez obtenir des prédictions des deux modèles
6. ✅ Vous pouvez voir les différences de performance

---

## 📝 Résumé

Ce projet vous enseigne comment :
- Compresser de grands modèles IA pour le déploiement en bord
- Comparer les performances des modèles de manière interactive
- Comprendre les compromis entre taille, vitesse et précision
- Construire des applications ML pratiques

**Commencez par le notebook pour entraîner les modèles, puis utilisez l'application Flask pour les comparer !**

---

**Bon Apprentissage ! 🚀**
