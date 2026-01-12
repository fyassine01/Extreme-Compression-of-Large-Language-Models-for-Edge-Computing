# Explication Simplifiée du Notebook : Compression de Modèles LLM

Voici une explication étape par étape du code contenu dans votre notebook pour la compression de modèles de langage (LLM).

**Objectif global :** Prendre un "gros" modèle d'intelligence artificielle (le **Professeur**) et créer une version beaucoup plus petite et rapide (l'**Étudiant**) capable de fonctionner sur des appareils limités (comme des téléphones ou des objets connectés), tout en gardant une bonne performance.

---

## 1. Installation et Préparation (Setup)
Le code commence par installer les "outils" nécessaires. C'est comme préparer son établi avant de bricoler.

*   **Ce que fait le code :** Il installe des bibliothèques clés comme `transformers` (pour manipuler les modèles de langage), `torch` (le moteur de calcul), et `onnx` (pour l'optimisation finale).
*   **Données :** Il télécharge le jeu de données **SST-2**, qui contient des phrases de critiques de films. Le but est de classer ces phrases en "positives" ou "négatives".

## 2. Le Professeur : Fine-tuning (Ajustement)
Avant de compresser, il faut un modèle de référence performant.

*   **Concept :** On prend un modèle **BERT-base** (puissant mais lourd) et on l'entraîne spécifiquement sur notre tâche de classification de sentiments pour qu'il devienne un expert.
*   **Dans le code :** On utilise un `Trainer` pour entraîner ce modèle. C'est notre référence de qualité (Accuracy ~89%).

## 3. L'Étudiant : Distillation de Connaissances (Knowledge Distillation)
C'est la première étape majeure de compression.

*   **Concept :** Au lieu d'entraîner un petit modèle à partir de zéro, on lui demande d'imiter le comportement du "Professeur" (BERT). Le petit modèle (**DistilBERT**) n'essaie pas seulement de trouver la bonne réponse, mais aussi de
