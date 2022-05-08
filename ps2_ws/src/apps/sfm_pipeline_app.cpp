#include <iostream>

#include "sfm_pipeline.hpp"

std::vector<Eigen::Matrix3d> fundamental_matrices;
std::vector<Eigen::MatrixXd> matches_subset;

double focal_length = 719.5459;

void data_loaded(void) {

    Eigen::Matrix3d f_tmp1;
    f_tmp1 << -4.04723691e-08, -2.55896526e-07, -4.93926104e-05,
              -1.34663426e-06,  1.96045499e-07, -5.56818858e-03,
               6.19417340e-04,  5.59528844e-03, -5.23679608e-04;
    Eigen::Matrix3d f_tmp2;
    f_tmp2 << 9.70161499e-08,  1.26743601e-06, -1.54506132e-04,
              4.83945527e-07,  3.25985186e-09,  5.60754818e-03,
             -4.74956177e-04, -5.47792282e-03,  2.26098145e-02;
    Eigen::Matrix3d f_tmp3;
    f_tmp3 << 6.75705582e-08, -1.09021840e-06,  5.93224732e-04,
              3.23134307e-06,  6.38757150e-08,  5.80151966e-03,
             -1.27135584e-03, -5.48838112e-03,  7.93359941e-02;
    Eigen::Matrix3d f_tmp4;
    f_tmp4 << -2.66322904e-08,  3.22928035e-07, -1.79473476e-04,
               1.98701325e-06, -4.02694866e-07,  6.93923582e-03,
              -5.76424460e-04, -7.13155017e-03, -3.92654749e-02;
    fundamental_matrices.push_back(f_tmp1);
    fundamental_matrices.push_back(f_tmp2);
    fundamental_matrices.push_back(f_tmp3);
    fundamental_matrices.push_back(f_tmp4);

    std::ifstream file_;
    file_.open("/home/euvill/Desktop/cs231a_ws/ps2_ws/data/statue/matches_subset.txt");
    std::string tmp;
    int index = 0;

    while (std::getline(file_, tmp)) {
        static bool flag_find_one_row = false;
        static std::vector<double> number;
        static int row_index = 0;
        static bool flag_resize = false;
        if (tmp == "begin") {
            row_index = 0;
            number.clear();
            flag_find_one_row = false;

            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> tmp_zero;
            matches_subset.push_back(tmp_zero);

            flag_resize = true;
        }
        else if (tmp == "end")
        {
            index = index + 1;
        }
        else
        {
            std::string str;
            for (int i = 0; i < tmp.size(); ++i) {
                if (tmp[i] == '[') {
                    flag_find_one_row = false;
                }else if (tmp[i] == ']') {
                    flag_find_one_row = true;
                    if(str.size() != 0) {
                        number.push_back(std::stod(str));
                        str.clear();
                    }
                    if(flag_resize) {
                        matches_subset[index].resize(4, number.size());
                        flag_resize = false;
                    }
                }else {
                    if(tmp[i] == ' ') {
                        if(str.size() != 0) {
                            number.push_back(std::stod(str));
                            str.clear();
                        }
                    }else {
                        str.push_back(tmp[i]);
                    }
                }
            }

            if(str.size() != 0) {
                number.push_back(std::stod(str));
                str.clear();
            }
    
            if (flag_find_one_row) {
                for (int i = 0; i < matches_subset[index].cols(); ++i) {
                    matches_subset[index](row_index ,i) = number[i];
                }
                number.clear();
                row_index = row_index + 1;
                flag_find_one_row = false;
            }
        }
    }

    file_.close();
}

int main(void) {
    
    data_loaded();

    std::vector<Frame> frames;

    int N = matches_subset.size();

    for (int i = 0; i < N; ++i) {
        
        Frame frame(matches_subset[i], focal_length, fundamental_matrices[i], 480, 640);

        bundle_adjustment(frame);

        frames.push_back(frame);

    }
    

}