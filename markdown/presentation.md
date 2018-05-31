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
target="_blank">MPIIGaze Dataset</a>. Ωστόσο υπάρχουν κι'άλλα dataset, όπως το 
<a href="https://www.idiap.ch/dataset/eyediap" target="_blank">Eyediap</a> και το
<a href="http://www.hci.iis.u-tokyo.ac.jp/datasets/" target="_blank">Multiview Dataset</a>.

* Οι αρχικές εικόνες έχουνε κανονικοποιηθεί με τέτοιο τρόπο, ώστε να εξετάζονται όλες οι εικόνες υπό τις ίδιες συνθήκες(πχ.φωτισμός). Επίσης κάθε μάτι εξετάζεται ανεξάρτητα από το άλλο.

* Τα δεδομένα που έχουμε στην διάθεση μας είναι:

	1. Οι εικόνες e του κάθε ματιού με διαστάσεις (W,H) = (60,36)
	2. Ηead Pose(2d διάνυσμα σε πολικές συντεταγμένες). Κάθε μάτι έχει το δικό του Head Pose 
	3. Gaze(2d διάνυσμα επίσης σε πολικές) το όποιο προσπαθούμε να κάνουμε predict. Για κάθε μάτι προβλέπουμε διαφορετικό Gaze

* Για τον αλγόριθμο Random Forest, κάνουμε reshape τις εικόνες των ματιών
  από (W,H) = (60,36) σε (15,9) τόσο για το training, όσο και για το testing  



## Υλοποίηση Αλγορίθμου

* Για την υλοποίηση του  αλγορίθμου, βασίστηκα στην αρχική υλοποίηση του Breiman[^1], κάνοντας κάποιες αλλαγές στον τρόπο που διαλέγουμε τα features κατά το
  split. Οι αλλαγές αυτές γίνανε σύμφωνα με το [^2]. 



## Ομαδοποίηση των δεδομένων με βάση τα Head Poses

* Για την υλοποίηση του  αλγορίθμου, αρχικά ομαδοποιούμε τα training samples σε
__P pose clusters__, με βάση το Head Pose

* Μεγαλύτερο αριθμός από Clusters δίνει 
__μεγαλύτερη ομοιότητα__ στις τιμές των Head Poses




## Κατασκευή του δάσους μέσα από Regression Decision Trees

* Δημιουργούμε τόσα δέντρα, όσα και τα Pose Clusters, δηλαδή P

* Κάθε δέντρο παίρνει training data από τα R-nearest Clusters. Δηλαδή
  γειτονικά Clusters δίνουν training data στο δέντρο( τα Clusters δηλαδή με
τα κοντινότερα Head Poses ) 

* Κάθε Cluster δίνει σε ένα δέντρο την τετραγωνική ρίζα των δειγμάτων που
  διαθέτει

* Ως ολικό error παίρνουμε το μέσο gaze error από όλα τα regression trees 


<div id="foto" style="text-align: center;">
   <img src="rnearest.jpeg" width="400" alt="foto1">
</div>


## Πώς εκπαιδεύεται το κάθε δέντρο


* Σε κάθε κόμβο ενός δέντρου, προσπαθούμε να "μάθουμε" συναρτήσεις της μορφής 

			f = px1 - px2

       όπου px1, px2 είναι οι Gray τιμές από 2 pixel της eye Image (W=15,H=9). 

* Τα pixels αυτά "μαθαίνονται" μέσα από το training. Επίσης προσπαθούμε να
  "μάθουμε" το βέλτιστο threshold τ για κάθε κόμβο, όπου:

	a. αν f < τ, τότε το training sample κατευθύνεται στο αριστερό υποδέντρο
	b. αν f >= τ, τότε κατευθύνεται στο δεξιό υποδέντρο


* Ο αλγόριθμος με τον οποίο υπολογίζουμε ποια είναι τα βέλτιστα pixels και το
  βέλτιστο threshold για το split σε κάθε κόμβο είναι το __residual sum of squares__
  

* Το minimun sum of squares προκύπτει από τον  min(typos athroismatos edw)




## Πώς γίνεται το testing


* Μόλις θέλουμε να ελέγξουμε ένα testing sample, δεν το στέλνουμε σε όλα τα
  δέντρα, αλλά στα R-nearest δέντρα με βάσει το head pose

* Τότε υπολογίζουμε το average σφάλμα σε όλα τα regression δέντρα 




## Aναφορές σε βιβλιογραφίες/δημοσιεύσεις

[^1]: Breiman, L., Friedman, J.,Olshen, R., and Stone, C. [1984] Classification and Regression Trees,  Wadsworth
[^2]: Y. Sugano, Y. Matsushita, and Y. Sato. Learning-by-synthesis for appearance-based 3d gaze estimation.