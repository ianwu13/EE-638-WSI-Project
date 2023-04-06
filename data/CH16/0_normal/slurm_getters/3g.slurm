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


echo "3g"

wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_081.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_081.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_083.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_083.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_085.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_085.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_086.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_086.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_087.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_087.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_088.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_088.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_089.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_089.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_091.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_091.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_093.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_093.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_095.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_095.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_096.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_096.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_098.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_098.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_100.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_100.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_101.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_101.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_103.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_103.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_106.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_106.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_107.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_107.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_109.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_109.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_111.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_111.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_112.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_112.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_115.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_115.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_118.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_118.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_119.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_119.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_120.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_120.tif"

echo "COMPLETE"
