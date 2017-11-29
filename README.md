VIRAGO stands for VIRAGO Interferometric Resonance Analysis for Globular Objects

The program is designed to count virus particles (virions) captured by antibodies on a silica-silicon substrate (in the form of a 10-16 mm chip). This substrate is designed to take advantage of the property of Interferometric Resonance (https://en.wikipedia.org/wiki/Interferometry). Essentially, virions are too small to be seen by traditional light microscopes, so typically electron microscopy (EM) is employed to study virion morphology. EM has several drawbacks, including sample preparation and expense.

This instrument that generates the data for which is called an IRIS - Interferometric Resonance Imaging Sensor- and was developed at Boston University in collaboration with nanoView Diagnostics.

********
Finally, I have uploaded the necessary dataset for using the VIRAGO software! It can be found here:
https://drive.google.com/file/d/1bzsoaAuBEe_Y4BqxaFcfN_rZ5oaCDZJq/view?usp=sharing
It is 3.23 GB in size, compressed as a 7zip archive. Most unarchivers should be able to unpack it to a size of 4.78 GB
Once unpacked, you will find it consists mostly of portable graymaps, which are images of antibody spots covalently bound to the silica chip captured by the IRIS. 
This chip (ID = tCHIP008) contains a grid-array of 6 different types of antibody, 3 of which recogize Ebola Virus (EBOV), and 3 that do not, serving as negative controls. More details about the chip can be found in the lone XML file within the archive.

This particular experiment was performed on November 2, 2017.
The experiment was performed over a course of about 30 minutes. During this time, saline solution containing a mostly-harmless EBOV mimic (VSV-EBOV) flowed over the surface of the chip at a rate of about 5 microliters per minute. 
The VSV-EBOV was at a concentration of 1 x 10^6 PFU/mL (https://en.wikipedia.org/wiki/Plaque-forming_unit), a typical concentration for virological research.
As VIRAGO scans through the the image files, which are in chronological order you will see an increase in the number of particles recognized on the antibody spots that recognize EBOV, and little change on the ones that do not. 
If all works well, VIRAGO will output a series of graphs and images showing the binding of the particles increasing, and histograms showing the relative size (as measured by percent contrast) of the particles. 
********
