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


echo "4g"

wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_24.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_24.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_25.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_25.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_26.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_26.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_27.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_27.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_28.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_28.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_29.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_29.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_30.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_30.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_31.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_31.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_32.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_32.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_33.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_33.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_34.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_34.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_35.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_35.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_36.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_36.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_37.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_37.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_38.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_38.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_39.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_39.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_40.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_40.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_41.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_41.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_42.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_42.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_43.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_43.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_44.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_44.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_45.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_45.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_46.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_46.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_47.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_47.tif"

echo "COMPLETE"
