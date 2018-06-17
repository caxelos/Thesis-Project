<html>
<body style="background: url(Downloads/gaze1_colorized.jpg) no-repeat center center fixed;
  -webkit-background-size: cover;
  -moz-background-size: cover;
  -o-background-size: cover;
  background-size: cover;">
</body>
</html>



<!-- background="Downloads/gaze1_colorized.jpg"; -->

---
title: Gaze prediction με χρήση Regression Random Forests
output:
  html_notebook:
    css: /home/trakis/format.css
---


<!---
%A Little Data Analysis
-->
Author: Christos Axelos
Date: May 30, 2018



## Σκοπός Ειδικού Θέματος

Πειραματική εξέταση του Αλγορίθμου __Random Forests__ στο πρόβλημα του Gaze Recognition


## Δεδομένα

* Ως δεδομένα επέλεξα το <a href="https://www.mpi-inf.mpg.de/de/abteilungen/computer-vision-and-multimodal-computing/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild-mpiigaze/"
target="_blank">MPIIGaze Dataset</a> [^3]. Ωστόσο υπάρχουν κι'άλλα dataset, όπως το <a href="https://www.idiap.ch/dataset/eyediap" target="_blank">Eyediap</a> και το <a href="http://www.hci.iis.u-tokyo.ac.jp/datasets/" target="_blank">Multiview Dataset</a> [^2].


* Οι αρχικές εικόνες έχουνε κανονικοποιηθεί με τέτοιο τρόπο, ώστε να εξετάζονται όλες οι εικόνες υπό τις __ίδιες συνθήκες__ .Επίσης κάθε μάτι εξετάζεται __ανεξάρτητα__ από το άλλο.


* Τα δεδομένα που έχουμε στην διάθεση μας είναι:

	1. Οι εικόνες __e__ του κάθε ματιού με διαστάσεις (W,H) = (60,36)

	2. __Ηead Pose__, 2d διάνυσμα γωνιών σε radians(γωνία Theta και γωνία
Phi)

	3. __Gaze__(2d διάνυσμα επίσης σε radians) το όποιο προσπαθούμε να κάνουμε predict. Κάθε μάτι γίνεται predict __ανεξάρτητα__ από το άλλο

	4. Η γωνία __Theta__ εκφράζει την οριζόντια θέση του κεφαλιού. Για
παράδειγμα αν το κεφάλι έχει προσανατολισμό  προς τα __δεξιά__, θα έχει
__θετική__ τιμή, ενώ αν κοιτάει προς τα __αριστερά__, θα έχει __αρνητική__.

	5. Η γωνία __Phi__ λειτουργεί σαν την Theta, αλλά για τον κάθετο άξονα.
Για παράδειγμα, αν το κεφάλι έχει προσανατολισμό προς τα __πάνω__, θα έχει
__θετική τιμή__, ενώ αν κοιτάει προς τα __κάτω__, θα έχει __αρνητική__

	6. Και οι 2 αυτές γωνίες κυμαίνονται στο διάστημα [-30, +30] σε
__μοίρες__


* Για τον αλγόριθμο Random Forest, κάνουμε __reshape__ τις εικόνες των ματιών
  από (W,H) = (__60,36__) σε (__15,9__) τόσο για το __training__, όσο και για το __testing__  





## Υλοποίηση Αλγορίθμου

* Για την υλοποίηση του  αλγορίθμου, βασίστηκα στην αρχική υλοποίηση του Breiman[^1], κάνοντας κάποιες αλλαγές στον τρόπο που διαλέγουμε τα __features__ κατά το split





## Ομαδοποίηση των δεδομένων με βάση τα Head Poses

* Για την υλοποίηση του  αλγορίθμου, αρχικά ομαδοποιούμε τα training samples σε __P pose clusters__, με βάση το __Head Pose__


* Κάθε Cluster έχει ένα __κέντρο__, το οποίο αποτελείται από ένα διάνυσμα
  (__theta, phi__)

* Για να θεωρηθεί ένα διανύσμα (__theta, phi__) ως κέντρο ενός Cluster, θα πρέπει να __μην απέχει__ απόσταση μικρότερη από Χ από τα ήδη υπάρχοντα κέντρα(πχ στο παρακάτω σχήμα χρησιμοποιώ __Χ=0.08__ και δημιουργούνται __106 Clusters__).

* Όσο __μικρότερο__ το Χ, τόσο πιο __πολλά__ Clusters δημιουργούνται



<div id="foto" style="text-align: center;">
   <img src="visualization.jpg"  alt="foto1">
</div>





## Κατασκευή του δάσους μέσα από Regression Decision Trees

* Χρησιμοποιώ την __bootstrap__ διαδικασία, επιλέγοντας τυχαία inputs

* Δημιουργούμε τόσα __δέντρα__, όσα και τα __Pose Clusters__, δηλαδή P

* Κάθε δέντρο παίρνει training data από τα __R-nearest Clusters__. Δηλαδή R Clusters
  με τα __κοντινότερα__ Head Poses


* Ως __predict__ παίρνουμε το __μέσο prediction__ από όλα τα δέντρα

* Ως __error__ παίρνουμε το __μέσο gaze error__ από όλα τα regression trees. Καταγράφουμε ωστόσο και την __τυπική απόκλιση__



<div id="foto" style="text-align: center;">
   <img src="rnearest.jpeg" width="400" alt="foto1">
</div>





## Πώς εκπαιδεύεται το κάθε δέντρο


* Σε κάθε κόμβο ενός δέντρου, προσπαθούμε να μάθουμε __συναρτήσεις__ της μορφής

$$
    f = px1 - px2
$$




* Τα px1, px2 είναι οι __Gray__ τιμές από 2 pixel της eye Image (W=15,H=9).

* Τα __pixels__ αυτά μαθαίνονται μέσα από το training. Επίσης προσπαθούμε να
  "μάθουμε" το __βέλτιστο threshold τ__ για κάθε κόμβο, όπου:

	a. αν $$ __f < τ__ $$, τότε το training sample κατευθύνεται στο __αριστερό__ υποδέντρο
	b. αν $$ __f >= τ__ $$, τότε κατευθύνεται στο __δεξιό__ υποδέντρο


* Ο αλγόριθμος με τον οποίο υπολογίζουμε ποια είναι τα βέλτιστα pixels και το
  βέλτιστο threshold για το split σε κάθε κόμβο είναι το __residual sum of squares__

  $$
  \begin{align}
  \sum_{i=0}^n i^2 &= \frac{(n^2+n)(2n+1)}{6} \\
  y &= mx+c
  \end{align}
  $$


* Παρακάτω φαίνεται ένα στιγμιότυπο ενός δέντρου

    (δείξε εικόνα εδώ)


* Ο τρόπος μάθησης των στοιχείων διαχωρισμού περιγράφεται παρακάτω:


	1. Για κάθε δυνατό ζευγάρι pixel(px1,px2)
		2. Για κάθε threshold
			3. Υπολόγισε το rightError= sum of squares error στο δεξί υποδέντρο  
			4. Υπολόγισε το leftError για το αριστερό υποδέντρο
			5. Error = rightError + leftError
			6. Αν Error < minError
				7. minError = Error;
				8. minPx1 = px1;
				9. minPx2 = px2;
				10. minThreshold = threshold;


* Οπότε έτσι μαθαίνουμε τα  minPx1, minPx2, minThreshold κάθε κόμβου





## Πώς γίνεται το testing


* Μόλις θέλουμε να ελέγξουμε ένα testing sample, δεν το στέλνουμε σε όλα τα
  δέντρα, αλλά στα R-nearest δέντρα με βάσει το head pose

* Τότε υπολογίζουμε το average σφάλμα σε όλα τα regression δέντρα, καθώς και
  το Standar Deviation





## Aναφορές σε βιβλιογραφίες/δημοσιεύσεις

[^1]: Breiman, L., Friedman, J.,Olshen, R., and Stone, C. [1984] Classification and Regression Trees,  Wadsworth
[^2]: Y. Sugano, Y. Matsushita, and Y. Sato. Learning-by-synthesis for appearance-based 3d gaze estimation.
[^3]: Z. Zhang, Y.Sugano, M.Fritz, A. Bulling [2015] Appearance-Based Gaze Estimation in the Wild
