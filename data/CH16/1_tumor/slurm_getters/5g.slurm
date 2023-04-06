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


echo "5g"

wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_48.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_48.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_49.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_49.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_50.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_50.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_51.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_51.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_52.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_52.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_53.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_53.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_54.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_54.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_55.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_55.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_56.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_56.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_57.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_57.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_58.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_58.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_59.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_59.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_60.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_60.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_61.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_61.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_62.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_62.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_63.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_63.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_64.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_64.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_65.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_65.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_66.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_66.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_67.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_67.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_68.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_68.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_69.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_69.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_70.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_70.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_71.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_71.tif"

echo "COMPLETE"
