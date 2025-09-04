# Implementation of Regression Forrests algorithm from Scratch in C and Matlab languages


<html>
<body style="background: url(/home/oly/MPIIGaze/markdown/gazeview.jpg) no-repeat center center fixed;
  -webkit-background-size: cover;
  -moz-background-size: cover;
  -o-background-size: cover;
  background-size: cover;">
</body>
</html>



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

* Για να θεωρηθεί ένα διανύσμα (__theta, phi__) ως κέντρο ενός Cluster, θα πρέπει να __μην απέχει__ απόσταση μικρότερη από __Χ__ από τα ήδη υπάρχοντα κέντρα(πχ στο παρακάτω σχήμα χρησιμοποιώ __Χ=0.08__ και δημιουργούνται __106 Clusters__).

* Όσο __μικρότερο__ το Χ, τόσο πιο __πολλά__ Clusters δημιουργούνται



<div id="foto" style="text-align: center;">
   <img src="centers.jpg"  alt="foto1">
   <figcaption><i>Διάγραμμα που απεικονίζει τα  <b>Head Poses</b> όλων των σημείων του Training Phase. Με <b>πράσινο</b> χρώμα απεικονίζονται τα κέντρα των Clusters, ενώ με <b>μπλέ</b> χρώμα τα υπόλοιπα σημεία. Η παραπάνω εικόνα χρησιμοποιεί <b>44640</b> training δείγματα, ενώ τα <b>κέντρα</b> απέχουν μεταξύ τους απόσταση <b>μεγαλύτερη</b> των <b>0.03 radians</b>(1.718873) μοίρες)</i></figcaption>
</div>




## Κατασκευή του δάσους μέσα από Regression Decision Trees

* Χρησιμοποιώ την __bootstrap__ διαδικασία, επιλέγοντας τυχαία inputs

* Δημιουργούμε τόσα __δέντρα__, όσα και τα __Pose Clusters__, δηλαδή P

* Κάθε δέντρο παίρνει training data από τα __R-nearest Clusters__. Δηλαδή R Clusters
  με τα __κοντινότερα__ Head Poses

* Ως __error__ παίρνουμε το __μέσο gaze error__ από όλα τα regression trees.



<div id="foto" style="text-align: center;">
   <img src="visualization.jpeg" width="400" alt="foto1">
   <figcaption><i>Παράδειγμα όπου <b>γειτονικά Clusters</b> συνεισφέρουν στην κατασκευή ενός δέντρου. Στα Clusters ανοίκουν δείγματα με <b>παρόμοια Head Poses</b></i></br></figcaption>
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


* Ο αλγόριθμος με τον οποίο υπολογίζουμε ποια είναι τα __βέλτιστα pixels__ και το __βέλτιστο threshold__ για το split σε __κάθε κόμβο__ του δέντρου είναι το __minimum residual sum of squares__


$$
\begin{align}
 error =
 \sum_{\substack{i:f_{j}\lt{thres}}}^{nleft}  (g_{i} - \hat{ m_{left} } )^2 + \sum_{\substack{i:f_{j}\ge{thres}}}^{nright} (g_{i} - \hat{ m_{right} } )^2

\end{align}
$$

* Τα $nleft$ και $nright$ είναι ο __αριθμός__ των δειγμάτων που θα είχε κάθε υποδέντρο, σε περίπτωση που γινόταν το split με βάση τα $px1$,$px2$,$thres$


* Τα  $\hat{ m_{right} }$ και $\hat{ m_{left} }$ είναι η __μέση τιμή__ των gazes που ανήκουν στο __δεξί__ και __αριστερό__ υποδέντρο


* Διαλέγουμε τα $px1$,$px2$,$thres$ που __ελαχιστοποιούν__ το παραπάνω άθροισμα

<div id="foto" style="text-align: center;">
   <img src="stigmiotupo.png" alt="foto1">
    <figcaption><i>Στιγμιότυπο υποδέντρου <b>10 δειγμάτων</b>. Ανάλογα με τις τιμές των <b>Pixel</b> του δείγματος, το τελευταίο θα οδηγηθεί σε έναν <b>τερματικό κόμβο</b>(φύλλο)</i></br></br> </figcaption>
</div>


* Ο τρόπος μάθησης των στοιχείων διαχωρισμού περιγράφεται παρακάτω:


	1. _Για κάθε δυνατό ζευγάρι pixel(px1,px2)_
		2. _Για κάθε threshold_
			3. _Υπολόγισε το rightError= sum of squares error στο δεξί υποδέντρο_  
			4. _Υπολόγισε το leftError για το αριστερό υποδέντρο_
			5. _Error = rightError + leftError_
			6. _Αν Error < minError_
				7. _minError = Error_
				8. _minPx1 = px1_
				9. _minPx2 = px2_
				10. _minThreshold = threshold_


* Οπότε έτσι μαθαίνουμε τα  minPx1, minPx2, minThreshold κάθε κόμβου





## Πώς γίνεται το testing


* Μόλις θέλουμε να ελέγξουμε ένα testing sample, δεν το στέλνουμε σε όλα τα
  δέντρα, αλλά στα __R-nearest δέντρα__ με βάσει το head pose

* Τότε υπολογίζουμε το __average error__ από τα R-nearest regression δέντρα.

* Μας ενδιαφέρει ωστόσο να γνωρίζουμε και την __τυπική απόκλιση__(standard deviation), για να βλέπουμε __πόσο κοντά__ είναι οι προβλέψεις μας στο __mean error__


## Πειραματική Αξιολόγηση Αλγορίθμου



* Κατά την αναλυτική αξιολόγηση του Αλγορίθμου θα πρέπει να απαντήσουμε τα εξής ερωτήματας:


  1. Ποιός είναι ο βέλτιστος __αριθμός__ από __Clusters__ ή ισοδύναμα ποιά θα είναι η __μικρότερη δυνατή απόσταση__ μεταξύ 2 κέντρων

  2. Ποιός είναι ο βέλτιστος __αριθμός δειγμάτων__ εκπαίδευσης

  3. Ποιός είναι ο βέλτιστος __αριθμός γειτόνων__ κάθε Cluster

  4. Κατά τη διαδικασία δημιουργίας υποδέντρων σε ένα δέντρο, πόσες __μεταβλητές διαχωρισμού__ χρησιμοποιούμε?

  4. Πόσο __βάθος__ πρέπει να έχει το κάθε δέντρου

  5. Αν σε ένα δέντρο καταλήξουμε σε φύλλο όπου υπάρχουν  __παραπάνω από ένα__ δείγματα, πώς προσδιορίζεται η 2d-gaze του συγκεκριμένου δέντρου



##### Ερώτημα 1ο: Εύρεση μικρότερης απόστασης Κέντρων/Αριθμός Clusters


* Επειδή δεν έχουμε υπολογίσει ακόμα τον __βέλτιστο αριθμό γειτόνων__, θεωρώ αρχικά πως __R = 5__ γείτονες.

* Επίσης, για να μειώσω τον χρόνο εκπαίδευσης, χρησιμοποιώ αρχικά __10,000 training samples__

* Ο αριθμός των __μεταβλητών διαχωρισμού__ που θα χρησιμοποιούσαμε είναι WIDTH*HEIGHT*THRESHOLD_RANGE = 15*16*255. Επειδή ο αριθμός αυτός είναι υπερβολικά μεγάλος, σύμφωνα με το paper του Breiman θα χρησιμοποιήσουμε την __τετραγωνική ρίζα__ αυτού του αριθμού

* Τέλος, αν ένα φύλλο ενός δέντρου περιέχει __παραπάνω από ένα__ training samples, υπολογίζω απλώς τον __μέσο όρο__ των __2-d gazes__ και προκύπτει ένα 2d-gaze διάνυσμα

* Δεν περιορίζουμε το βάθος του κάθε δέντρου

* Με βάση τα παραπάνω, υπολογίζουμε το __mean error__ και το __standard deviation__



## Aναφορές σε βιβλιογραφίες/δημοσιεύσεις

[^1]: Breiman, L., Friedman, J.,Olshen, R., and Stone, C. [1984] Classification and Regression Trees,  Wadsworth
[^2]: Y. Sugano, Y. Matsushita, and Y. Sato. Learning-by-synthesis for appearance-based 3d gaze estimation.
[^3]: Z. Zhang, Y.Sugano, M.Fritz, A. Bulling [2015] Appearance-Based Gaze Estimation in the Wild
