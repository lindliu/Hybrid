#define M             1000
#define Q             1      // Careful with mod(PotRadius/Q)
#define Nodes         M*Q

#define MaxM          M+1
#define MaxQ          Q+1
#define MaxSites      Nodes+1 

#define PotRadius     20       // Must use a radius bigger than Q
#define Cont          0        // 1 to cont, 0 to start fresh
#define Hopf          0
#define Scalar        0
#define Saddle        1

short  initialscheme = 1;

double X_0        = 1.0;     double Y_0 = 1.0;
double timestep   = 0.001;
double final_time = 100.0;

#if Hopf==1
   double tau_I =  1.0, tau_c = 5.0;
   double J_0   =  1.0, beta  = 5.0;

   double z          =   0.5;
   double b          =   4.0;
   double barsigma   =   0.5;
   double omega      =   1.0;
   double gam        =   0.9;

   double extpotconst1 = .5, extpotconst2 = -1.; 
#endif
#if Saddle==1
   double tau_I =  1.0, tau_c = 1.0;
   double J_0   =  1.0, beta  = 0.01;

   double gam   = -.05;
   double z = 0.5;
   double b = 1.0;
   double barsigma = 0.01;
   double omega = 1.0;

   double extpotconst1 = 5., extpotconst2 = -1.; 
#endif
#if Scalar==1
   double tau_I =  1.0, tau_c = 0.1;
   double J_0   =  1.0, beta  = 2.0;

   double gam   = -.025;

   double extpotconst1 = -.5, extpotconst2 = 1.; 
#endif
