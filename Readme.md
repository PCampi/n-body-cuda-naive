# Simulazione n-body

La procedura di simulazione è la seguente:

1. scegli un numero di corpi, compreso tra 1024 - 2048 - 4096 - 8192 - 16384
2. per ogni numero di corpi, scegli un numero di iterazioni da portare a termine per la media e la varianza - diciamo 30
3. collegati in ssh alla macchina: `ssh pmcampi@s-nomadis-06.mlab.disco.unimib.it` ed entra nella cartella `nbody -> data`
4. digita `python run.py N_CORPI N_ITER_PER_MEDIA`
5. go for a coffee ;)
6. trova i risultati in results_static.csv


## nbody-cuda-naive

Questa versione è piuttosto stupida e impiega circa 52 secondi per completare un run con 16384 corpi.

Comunque, non è malaccio.

Il problema è questo: ho 3 kernels, che devono operare in sequenza, e li lancio sempre con 32 blocchi da 256 threads l'uno.

- `compute_a`: richiede 46 registri e non permette l'occupazione piena di uno SM, perché mi manda a pallino il numero massimo di registri per SM. In pratica, ho **solo** il 63% dei warps su uno SM occupati: anziché 64, solo 40 perché altrimenti sforo il numero di registri.
- `compute_v`: richiede 18 registri, ho occupazione piena degli SM.
- `integrate_position`: richiede 16 registri, ho occupazione piena degli SM.

Il collo di bottiglia è la procedura che calcola l'accelerazione sul corpo i-esimo data da tutti gli altri corpi j-esimi.


## nbody-cuda-optimized
