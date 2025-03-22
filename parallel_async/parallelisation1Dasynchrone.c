#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <mpi.h>
#include <time.h>

/* AUTHOR : AUTHORS : C. Bouillaguet, parallelised by T. Chenaille
   USAGE  : compile with -lm (and why not -O3)
            redirect the standard output to a text file
            gcc parallelisation1D.c -O3 -lm -o parallelisation1D
            ./parallelisation1D > steady_state.txt
            then run the indicated python script for graphical rendering
*/

/* one can change the matter of the heatsink, its size, the power of the CPU, etc. */
#define COPPER
#define MEDIUM   /* Slowest to fastest : CHALLENGE, NORMAL, MEDIUM, and FAST (for debugging) */
#define DUMP_STEADY_STATE

const double L = 0.15;      /* longueur (x) du radiateur (en mètres) */
const double l = 0.12;      /* largeur (y) du radiateur (en mètres) */
const double E = 0.008;     /* épaisseur (z) du radiateur (en mètres) */
const double watercooling_T = 20;   /* temperature of the fluid for water-cooling, (°C) */
const double CPU_TDP = 280; /* power dissipated by the CPU (W) */

/* dl: "spatial step" for simulation (m) */
/* dt: "time step" for simulation (s) */
#ifdef FAST
double dl = 0.004;
double dt = 0.004;
char mode[] = "FAST";
#endif

#ifdef MEDIUM
double dl = 0.002;
double dt = 0.002;
char mode[] = "MEDIUM";
#endif

#ifdef NORMAL
double dl = 0.001;
double dt = 0.001;
char mode[] = "NORMAL";
#endif

#ifdef CHALLENGE
double dl = 0.0001;
double dt = 0.00001;
char mode[] = "CHALLENGE";
#endif

/* sink_heat_capacity: specific heat capacity of the heatsink (J / kg / K) */
/* sink_density: density of the heatsink (kg / m^3) */
/* sink_conductivity: thermal conductivity of the heatsink (W / m / K) */
/* euros_per_kg: price of the matter by kilogram */
#ifdef ALUMINIUM
double sink_heat_capacity = 897;
double sink_density = 2710;
double sink_conductivity = 237;
double euros_per_kg = 1.594;
#endif

#ifdef COPPER
double sink_heat_capacity = 385;
double sink_density = 8960;
double sink_conductivity = 390;
double euros_per_kg = 5.469;
#endif

#ifdef GOLD
double sink_heat_capacity = 128;
double sink_density = 19300;
double sink_conductivity = 317;
double euros_per_kg = 47000;
#endif

#ifdef IRON
double sink_heat_capacity = 444;
double sink_density = 7860;
double sink_conductivity = 80;
double euros_per_kg = 0.083;
#endif

const double Stefan_Boltzmann = 5.6703e-8;  /* (W / m^2 / K^4), radiation of black body */
const double heat_transfer_coefficient = 10;    /* coefficient of thermal convection (W / m^2 / K) */
double CPU_surface;

void printArray(double local_R[], int n) {
    for (int i = 0; i < n; i++) {
        fprintf(stderr, "%f ", local_R[i]);
    }
    printf("\n");
}

/* 
 * Returns True if the CPU is in contact with the heatsink at the point (x,y).
 * This describes an AMD EPYC "Rome".
 */
static inline bool CPU_shape(double x, double y)
{
    x -= (L - 0.0754) / 2;
    y -= (l - 0.0585) / 2;
    bool small_y_ok = (y > 0.015 && y < 0.025) || (y > 0.0337 && y < 0.0437);
    bool small_x_ok = (x > 0.0113 && x < 0.0186) || (x > 0.0193 && x < 0.0266)
        || (x > 0.0485 && x < 0.0558) || (x > 0.0566 && x < 0.0639);
    bool big_ok = (x > 0.03 && x < 0.045 && y > 0.0155 && y < 0.0435);
    return big_ok || (small_x_ok && small_y_ok);
}

/* Returns the total area of the surface of contact between CPU and heatsink (in m^2) */
double CPU_contact_surface()
{
    double S = 0;
    for (double x = dl / 2; x < L; x += dl)
        for (double y = dl / 2; y < l; y += dl)
            if (CPU_shape(x, y))
                S += dl * dl;
    return S;
}

/* Retourne la nouvelle température de la cellule (i, j, k). Pour ce faire, un accès aux 6 cellules voisines est effectué
 * (voisines de gauche, de droite, du haut, du bas, du devant, du derrière) sauf si (i, j, k) est sur la surface extérieure (cas particulier). */
static inline double update_temperature_local(const double *T, int u, int n, int m, int o, int i, int j, int k, int start_layer, int end_layer, int rank, int size, double *B1, double *B2)
{
    /* Quantité d'énergie thermique devant être apporté à une cellule pour augmenter sa température de 1°C. */
    const double cell_heat_capacity = sink_heat_capacity * sink_density * dl * dl * dl; /* J.K */
    const double dl2 = dl * dl;
    double thermal_flux = 0;

    /* AXE X - Les voisins sur l'axe Z de la cellule courante sont toujours nécessairement dans la
    zone de travail du processus courant puisque notre découpage a été fait sur l'axe Y et non X.
    On peut donc y accéder directement sans aller les chercher dans les processus voisins.*/
    
    if (i > 0)
        thermal_flux += (T[u - 1] - T[u]) * sink_conductivity * dl; // Voisin x-1
    else {
        thermal_flux -= Stefan_Boltzmann * dl2 * pow(T[u], 4);
        thermal_flux -= heat_transfer_coefficient * dl2 * (T[u] - watercooling_T);
    }

    if (i < n - 1)
        thermal_flux += (T[u + 1] - T[u]) * sink_conductivity * dl; // Voisin x+1
    else {
        thermal_flux -= Stefan_Boltzmann * dl2 * pow(T[u], 4);
        thermal_flux -= heat_transfer_coefficient * dl2 * (T[u] - watercooling_T);
    }

    /* AXE Z - Les voisins sur l'axe Z de la cellule courante sont toujours nécessairement dans la
    zone de travail du processus courant puisque notre découpage a été fait sur l'axe Y et non Z.
    On peut donc y accéder directement sans aller les chercher dans les processus voisins.*/
    
    if (j > 0)
        thermal_flux += (T[u - n] - T[u]) * sink_conductivity * dl; // Voisin z-1
    else {
        /* Bottom cell: does it receive directly from the CPU ? */
        if (CPU_shape(i * dl, k * dl))
            thermal_flux += CPU_TDP / CPU_surface * dl2;
        else {
            thermal_flux -= Stefan_Boltzmann * dl2 * pow(T[u], 4);
            thermal_flux -= heat_transfer_coefficient * dl2 * (T[u] - watercooling_T);
        }
    }
    
    if (j < m - 1)
        thermal_flux += (T[u + n] - T[u]) * sink_conductivity * dl; // Voisin z+1
    else {
        thermal_flux -= Stefan_Boltzmann * dl2 * pow(T[u], 4);
        thermal_flux -= heat_transfer_coefficient * dl2 * (T[u] - watercooling_T);
    }

    /* AXE Y - Les ajustements sont faits seulement sur cet axe, puisque c'est le long
    de celui-ci que l'on a effectué notre découpe. Dans certains cas, on doit aller chercher
    les cellules voisines dans les zones de travail des processus voisins.*/
    
    /* Cas du voisin y-1 */
    if (k != start_layer){
        // Le voisin est dans la zone de travail du processus courant. On y accède directement.
        thermal_flux += (T[u - n * m] - T[u]) * sink_conductivity * dl;
    }
    else{
        // Le voisin est dans la zone de travail d'un autre processus. Il faut donc aller
        // le chercher dans recv_buffer_fromPrevProcess (appelé B1 dans cette fonction).
        thermal_flux += (B1[j * n + i] - T[u]) * sink_conductivity * dl;
    }
    
    /* Cas du voisin y+1 */
    if (k < o - 1){
    	if (k != end_layer) {
    	    // Le voisin est dans la zone de travail du processus courant. On y accède directement.
            thermal_flux += (T[u + n * m] - T[u]) * sink_conductivity * dl;
        }
        else{
            // Le voisin est dans la zone de travail d'un autre processus. Il faut donc aller
            // le chercher dans recv_buffer_fromNextProcess (appelé B2 dans cette fonction).
            thermal_flux += (B2[j * n + i] - T[u]) * sink_conductivity * dl;
        }
    }
    
    /* Cas ou la cellule est sur un bord du heatsink */
    else {
        thermal_flux -= Stefan_Boltzmann * dl2 * pow(T[u], 4);
        thermal_flux -= heat_transfer_coefficient * dl2 * (T[u] - watercooling_T);
    }

    /* Ajustement de la température en fonction du flux thermique */
    return T[u] + thermal_flux * dt / cell_heat_capacity;
}

/* Lance la simulation sur le k-ième plan zx.
 * 'local_index' est l'index de départ du k-ième plan zx dans les tableaux 'local_T' et 'local_R'. */
static inline void do_zx_plane_local(const double *T, double *R, int local_index, int n, int m, int o, int k, int start_layer, int end_layer, int rank, int size, double *B1, double *B2)
{
    // On ne modifie pas le plan y=0 : il est maintenu à une température constante par water-cooling.
    if (k == 0){return;}
    
    // On appelle la fonction 'update_temperature_local()' pour toutes les cellules du plan zx concerné.
    for (int j = 0; j < m; j++) {   // z
        for (int i = 0; i < n; i++) {   // x
            int u = local_index + j * n + i;
            R[u] = update_temperature_local(T, u, n, m, o, i, j, k, start_layer, end_layer, rank, size, B1, B2);
        }
    }
}

int main(int argc, char *argv[])
{
    CPU_surface = CPU_contact_surface();
    double V = L * l * E;
    int n = ceil(L / dl);
    int m = ceil(E / dl);
    int o = ceil(l / dl);
    int total_cells = n*m*o;
    
    /* Initialisation MPI pour la parallélisation */
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Status status;

    /* Affichage des caractéristiques de notre matériel et de notre simulation */
    int buffer_size = n * m;
    int buffer_amount = (size-1)*2;
    if (rank == 0){
    fprintf(stderr, "HEATSINK\n");
    fprintf(stderr, "\tDimension (cm) [x,y,z] = %.1f x %.1f x %.1f\n", 100 * L, 100 * l, 100 * E);
    fprintf(stderr, "\tVolume = %.1f cm^3\n", V * 1e6);
    fprintf(stderr, "\tWeight = %.2f kg\n", V * sink_density);
    fprintf(stderr, "\tPrice = %.2f €\n", V * sink_density * euros_per_kg);
    fprintf(stderr, "SIMULATION\n");
    fprintf(stderr, "\tGrid (x,y,z) = %d x %d x %d (%.1fMo)\n", n, o, m, 7.6293e-06 * n * o * m);
    fprintf(stderr, "\tThere are %d buffers (%.1fMo of data exchanged at each iteration)\n", buffer_amount, buffer_amount * 7.6293e-06 * buffer_size);
    fprintf(stderr, "\tdt = %gs\n", dt);
    fprintf(stderr, "CPU\n");
    fprintf(stderr, "\tPower = %.0fW\n", CPU_TDP);
    fprintf(stderr, "\tArea = %.1f cm^2\n", CPU_surface * 10000);}
    
    /* Répartition des couches (dimension o<=>y<=>l<=>k) */
    int layers_per_proc = o / size;  // Nombre de base de couches par processus
    int remainder = o % size;        // Couches supplémentaires à répartir
    int start_layer, end_layer;

    if (rank < remainder) {
        // Les premiers processus prennent une couche supplémentaire
        start_layer = rank * (layers_per_proc + 1);
        end_layer = start_layer + layers_per_proc;
    } else {
        start_layer = rank * layers_per_proc + remainder;
        end_layer = start_layer + layers_per_proc - 1;
    }
    int local_cell_count = (end_layer - start_layer + 1) * n * m;
    
    /* Initialisation des arrays stockant les températures de chaque cellule, en degrés Kelvin. */
    double *local_T = malloc(local_cell_count * sizeof(*local_T));
    double *local_R = malloc(local_cell_count * sizeof(*local_R));
    if (local_T == NULL || local_R == NULL) {
        perror("local_T or local_R could not be allocated");
        exit(1);
    }
    
    /* Déclaration des buffers d'envoi et de réception pour les couches frontalières */
    double *send_buffer_toPrevProcess = NULL;
    double *recv_buffer_fromPrevProcess = NULL;
    double *send_buffer_toNextProcess = NULL;
    double *recv_buffer_fromNextProcess = NULL;
    
    //Si le processus courant est le premier processus
    if ((rank == 0) && (size > 1)){
        send_buffer_toNextProcess = (double *)malloc(buffer_size * sizeof(double));
        recv_buffer_fromNextProcess = (double *)malloc(buffer_size * sizeof(double));
    }
    //Si le processus courant est le dernier processus
    if ((rank == (size-1)) && (size > 1)){
        send_buffer_toPrevProcess = (double *)malloc(buffer_size * sizeof(double));
        recv_buffer_fromPrevProcess = (double *)malloc(buffer_size * sizeof(double));
    }
    //Si le processus courant n'est ni le premier ni le dernier
    if ((rank != 0) && (rank != (size-1)) && (size > 1)){
        send_buffer_toPrevProcess = (double *)malloc(buffer_size * sizeof(double));
        //Si le processus ne traite qu'une seule couche, il envoie le meme buffer au processus suivant et précédent
        if (end_layer - start_layer == 0){
            send_buffer_toNextProcess = send_buffer_toPrevProcess;
        }
        //Sinon, il enverra deux buffers distincts
        else{
            send_buffer_toNextProcess = (double *)malloc(buffer_size * sizeof(double));
        }
        //Dans tous les cas, ses deux buffers de réception sont distincts
        recv_buffer_fromPrevProcess = (double *)malloc(buffer_size * sizeof(double));        
        recv_buffer_fromNextProcess = (double *)malloc(buffer_size * sizeof(double));
    }

    /* A l'état initial, le dissipateur est à la température du liquide de refroidissement */
    for (int i = 0; i < local_cell_count; i++) {
        local_R[i] = local_T[i] = watercooling_T + 273.15;
    }
    
    /* On initialise le contenu des buffers d'envoi et de réception de la même manière */
    if(size > 1){
    for (int i = 0; i < buffer_size; i++) {
        if(send_buffer_toPrevProcess != NULL){send_buffer_toPrevProcess[i] = watercooling_T + 273.15;}
        if(recv_buffer_fromPrevProcess != NULL){recv_buffer_fromPrevProcess[i] = watercooling_T + 273.15;}
        if(send_buffer_toNextProcess != NULL){send_buffer_toNextProcess[i] = watercooling_T + 273.15;}
        if(recv_buffer_fromNextProcess != NULL){recv_buffer_fromNextProcess[i] = watercooling_T + 273.15;}
    }}

    /* On allume le CPU et on lance la simulation jusqu'à ce qu'elle atteigne un état stationnaire */
    double t = 0;
    int n_steps = 0;
    int convergence = 0;
    double start_time, end_time;
    double start_comm_time, end_comm_time; 
    double total_comm_time = 0.0;
    int local_index, num_requests;
    
    
    /* On commence l'enregistrement du temps dans le processus maître uniquement. On le fait ici car
    on est sur le point d'entrer dans la boucle de simulation qui est l'endroit de calcul intense */
    if (rank == 0) {
        start_time = MPI_Wtime();
    }   

    /*  Boucle de simulation */
    MPI_Request recv_requests[2], send_requests[2];
    while (convergence==0) {

        int num_requests_recv = 0, num_requests_send = 0;

        // Initialisation des demandes de réception asynchrones
        if (size > 1 && n_steps != 0) {
            if (rank < size - 1) {
                start_comm_time = MPI_Wtime();
                MPI_Irecv(recv_buffer_fromNextProcess, n * m, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &recv_requests[num_requests_recv++]);
                end_comm_time = MPI_Wtime();
                total_comm_time += (end_comm_time - start_comm_time) ;
            }
            if (rank > 0) {
                start_comm_time = MPI_Wtime();
                MPI_Irecv(recv_buffer_fromPrevProcess, n * m, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &recv_requests[num_requests_recv++]);
                end_comm_time = MPI_Wtime();
                total_comm_time += (end_comm_time - start_comm_time) ;
            }
        }
        
        // Si n_step = 0, on peut mettre à jour les cellules frontalières directement
        if (n_steps == 0){
            for (int k = start_layer; k < (end_layer+1); k++) {
                local_index = (k - start_layer) * n * m;
                do_zx_plane_local(local_T, local_R, local_index, n, m, o, k, start_layer, end_layer, rank, size, recv_buffer_fromPrevProcess, recv_buffer_fromNextProcess);
            }
        }else{

        // Mise à jour des cellules non-frontalières
        for (int k = start_layer + 1; k < end_layer; k++) {
            local_index = (k - start_layer) * n * m;
            do_zx_plane_local(local_T, local_R, local_index, n, m, o, k, start_layer, end_layer, rank, size, NULL, NULL);
        }}

        // Attente de la réception de tous les buffers demandés
            for (int i = 0; i < num_requests_recv; i++) {
                MPI_Wait(&recv_requests[i], MPI_STATUS_IGNORE);
            }
            // Mise à jour des cellules frontalières avec les buffers reçus
            if (rank < size - 1) {
                   // Mise à jour des cellules frontalières avec le buffer du processus suivant
                   local_index = (end_layer - start_layer) * n * m;
                do_zx_plane_local(local_T, local_R, local_index, n, m, o, end_layer, start_layer, end_layer, rank, size, recv_buffer_fromPrevProcess, recv_buffer_fromNextProcess);
            } else if (end_layer - start_layer != 0) {
                local_index = (end_layer - start_layer) * n * m;
                do_zx_plane_local(local_T, local_R, local_index, n, m, o, end_layer, start_layer, end_layer, rank, size, NULL, NULL);
            }
            if (rank > 0) {
                  // Mise à jour des cellules frontalières avec le buffer du processus précédent
                do_zx_plane_local(local_T, local_R, 0, n, m, o, start_layer, start_layer, end_layer, rank, size, recv_buffer_fromPrevProcess, recv_buffer_fromNextProcess);
            } else {
                do_zx_plane_local(local_T, local_R, 0, n, m, o, start_layer, start_layer, end_layer, rank, size, NULL, NULL);
            }

        // Remplissage des buffers pour l'envoi
        if (size > 1) {
            if (rank == 0) {
                for (int f = 0; f < n * m; f++) {
                    send_buffer_toNextProcess[f] = local_R[end_layer * n * m + f];
                }
                MPI_Isend(send_buffer_toNextProcess, n * m, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &send_requests[num_requests_send++]);
            } else if (rank == size - 1) {
                for (int f = 0; f < n * m; f++) {
                    send_buffer_toPrevProcess[f] = local_R[f];
                }
                MPI_Isend(send_buffer_toPrevProcess, n * m, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &send_requests[num_requests_send++]);
            } else {
                for (int f = 0; f < n * m; f++) {
                    send_buffer_toPrevProcess[f] = local_R[f];
                    send_buffer_toNextProcess[f] = local_R[(end_layer - start_layer) * n * m + f];
                }
                MPI_Isend(send_buffer_toPrevProcess, n * m, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &send_requests[num_requests_send++]);
                MPI_Isend(send_buffer_toNextProcess, n * m, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &send_requests[num_requests_send++]);
            }
        }
        
        /* A chaque seconde (de temps fictif simulé), on teste la convergence et on print l'avancement. */
        if (n_steps % ((int)(1 / dt)) == 0) {
            double local_delta_T = 0;
            double local_max = -INFINITY;
            for (int u = 0; u < local_cell_count; u++) {
                local_delta_T += (local_R[u] - local_T[u]) * (local_R[u] - local_T[u]);
                if (local_R[u] > local_max)
                    local_max = local_R[u];
            } 
            
            if (size==1) {
                local_delta_T = sqrt(local_delta_T)/dt;
                fprintf(stderr, "t = %.1fs ; T_max = %.1f°C ; convergence = %g\n", t, local_max - 273.15, local_delta_T);
                if (local_delta_T < 0.1){convergence = 1;}
            }
            
            //Si size > 1, on ne fait pas la racine sur le local_delta_T ici mais sur le global_delta_T après communication
	    	else {
            	double global_delta_T = 0;
            	double global_max = 0;

            	// Utilisation de MPI_Allreduce pour obtenir la somme de local_delta_T et le maximum de local_max sur tous les processus
            	MPI_Allreduce(&local_delta_T, &global_delta_T, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            	MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

            	// Chaque processus calcule la moyenne globale de delta_T
            	global_delta_T = sqrt(global_delta_T)/dt;

            	if (rank == 0) {
                    // Rapport imprimé par le processus maître
                    fprintf(stderr, "t = %.1fs ; T_max = %.1f°C ; convergence = %g\n", t, global_max - 273.15, global_delta_T);
            	}

                // Vérification globale de la convergence (tous les processus doivent l'effectuer pour régler leur convergence)
                if (global_delta_T < 0.1){
                    convergence = 1;
                    if (rank == 0){
                        // Arrêt du chronomètre
                        end_time = MPI_Wtime();
                        fprintf(stderr, "\nTemps d'exécution des calculs : %f secondes", end_time - start_time);
                        fprintf(stderr, "\nRésultats obtenus pour %d processus (MPI-asynchrone) en mode %s\n\n", size, mode);
                    }
                    if (rank < size - 1) {
                        MPI_Recv(recv_buffer_fromNextProcess, n * m, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                    if (rank > 0) {
                        MPI_Recv(recv_buffer_fromPrevProcess, n * m, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                }
                }
        }       

        /* Les nouvelles températures sont dans local_R, on met à jour local_T en conséquence*/
        double *tmp = local_R;
        local_R = local_T;
        local_T = tmp;
        t += dt;
        n_steps += 1;   
    }
    
    // Libération des tableaux locaux
    free(local_T);
    free(local_R);
    // Libération des buffers d'envoi
    if (send_buffer_toPrevProcess != NULL) {
        free(send_buffer_toPrevProcess);
        if (send_buffer_toNextProcess == send_buffer_toPrevProcess) {send_buffer_toNextProcess = NULL;}
            send_buffer_toPrevProcess = NULL;}
    if (send_buffer_toNextProcess != NULL) {
        free(send_buffer_toNextProcess);
        send_buffer_toNextProcess = NULL;}
    // Libération des buffers de réception
    if (recv_buffer_fromPrevProcess != NULL) {free(recv_buffer_fromPrevProcess);}
    if (recv_buffer_fromNextProcess != NULL) {free(recv_buffer_fromNextProcess);}
    // Finalisation MPI et fin d'éxécution
    MPI_Finalize();
    exit(EXIT_SUCCESS);
    
    /* On a fini la boucle de simulation ! On prépare la sortie. */
    int *recvcounts = NULL, *displs = NULL;
    double *global_T = NULL;
    if (rank == 0) {
        // Préparation des tableaux nécessaires à la sortie
        recvcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        global_T = malloc(total_cells * sizeof(double));
    }

    /* Chaque processus envoie le nombre de cellules qu'il a traitées, 
    puis on rassemble les données de température */
    MPI_Gather(&local_cell_count, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        int total = 0;
        for (int i = 0; i < size; i++) {
            displs[i] = total;
            total += recvcounts[i];
        }
    }
    MPI_Gatherv(local_T, local_cell_count, MPI_DOUBLE, global_T, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

/* Inscription de l'état stationnaire dans le fichier passé en argument lors de l'exécution du programme C.
Si aucun fichier de sortie n'a été renseigné, l'affichage se fait dans la console.*/
#ifdef DUMP_STEADY_STATE
    if (rank == 0) {
        printf("###### STEADY STATE; t = %.1f\n", t);
        for (int k = 0; k < o; k++) {   // y
            printf("# y = %g\n", k * dl);
            for (int j = 0; j < m; j++) {   // z
                for (int i = 0; i < n; i++) {   // x
                    printf("%.1f ", global_T[k * n * m + j * n + i] - 273.15);
                }
                printf("\n");
            }
        }
        printf("\n");
        fprintf(stderr, "For graphical rendering: python3 rendu_picture_steady.py [filename.txt] %d %d %d\n", n, m, o);
        free(recvcounts);
        free(displs);
        free(global_T);
    }
      
#endif

/* Libération des allocations dynamiques et fin de l'exécution */

// Libération des tableaux locaux
free(local_T);
free(local_R);
// Libération des buffers d'envoi
if (send_buffer_toPrevProcess != NULL) {
    free(send_buffer_toPrevProcess);
    if (send_buffer_toNextProcess == send_buffer_toPrevProcess) {send_buffer_toNextProcess = NULL;}
    send_buffer_toPrevProcess = NULL;}
if (send_buffer_toNextProcess != NULL) {
    free(send_buffer_toNextProcess);
    send_buffer_toNextProcess = NULL;}
// Libération des buffers de réception
if (recv_buffer_fromPrevProcess != NULL) {free(recv_buffer_fromPrevProcess);}
if (recv_buffer_fromNextProcess != NULL) {free(recv_buffer_fromNextProcess);}
// Finalisation MPI et fin d'éxécution
MPI_Finalize();
exit(EXIT_SUCCESS);
}
