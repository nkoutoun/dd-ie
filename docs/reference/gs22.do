***********************************************************************************
* This code produces a within-interaction estimator of two time-varying variables *
* x and z in a fixed effects framework ("double demeaned interaction estimator"). *
* It compares the coefficient with the estimator obtained from a conventional FE- *
* approach and computes a hausman-test to detect a possible bias in this          *
* conventional estimator                                                          * 
***********************************************************************************
* To run smoothly, this ado needs the package "center"                            *  
* from http://fmwww.bc.edu/RePEc/bocode/cp                                        * 
***********************************************************************************
* The example presented below uses infant birth weight data (Abrevaya 2006) to    *
* estimate the interaction between smoking behaviour and maternal age on          *
* birthweight. However, any other data & variables can be defined in the first    * 
* section of the do-file. The code in its current form relies on continuously     * 
* scaled or dummy-variables as moderators                                         *
***********************************************************************************
set more off
version 13
*********************Data and Variables******************

*path to data
global data "http://www.stata-press.com/data/mlmus2/smoking"

*define unitvar
global i "momid"

*define timevar
global t "idx"

*define time-varyig interactors
local x "mage"
local z "smoke"

*define dependent variable
local y "birwt"

*define covariates
local w  "married"

use $data, clear
xtset $i $t
des
xtdes
preserve
center `y' `x' `z' `w', inplace // grand mean centering enables comparison of main effects across models

*********************Estimate Classic FE Interaction****************************
xtreg `y' c.`x'##c.`z' `w' , fe
*or via
generate int_`x'_`z'=`x'*`z'
tempvar sample
xtreg `y' `x' `z' int_`x'_`z' `w', fe
estimates store FE_IE
generate `sample'=1 if e(sample)==1

*******Estimate double demeaned (within-unit) interaction***********************
foreach var of varlist `x' `z'{
egen mean`var'=mean(`var') if `sample'==1, by ($i)
generate dm`var'=`var'-mean`var' 
}
replace int_`x'_`z'= dm`x' * dm`z'
qui xtreg `y' `x' `z' int_`x'_`z' `w', fe		
estimates store dd_IE		
di _n in gr "Fixed-effects regression with double-demeaned interaction xz" 
est replay dd_IE								

* Compare FE_IE ,dd_IE 
est tab FE_IE dd_IE	, b(%9.5f) se(%9.5f) p(%9.8f)	

*****Hausman-test on systematic differences between models with standard fe-****
*****and "real" within-estimator of interaction.********************************
noisily hausman dd_IE FE_IE		 
restore
