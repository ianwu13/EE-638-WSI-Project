#!/bin/bash
#SBATCH --account=ajiteshs_1045
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00

# Change to proper directory
if [[ "$PWD" =~ "slurm_getters" ]]
then 
    cd ..
else
    echo 'SLURM JOBS MUST BE RUN FROM "slurm_getters" DIRECTORY FOR THIS PROJECT'
    return -1
fi


echo "1g"

wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_003.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_003.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_005.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_005.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_006.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_006.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_007.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_007.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_009.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_009.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_012.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_012.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_014.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_014.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_015.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_015.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_017.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_017.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_018.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_018.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_019.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_019.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_020.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_020.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_022.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_022.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_023.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_023.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_024.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_024.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_025.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_025.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_028.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_028.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_031.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_031.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_032.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_032.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_034.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_034.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_035.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_035.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_036.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_036.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_037.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_037.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_039.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_039.tif"

echo "COMPLETE"
