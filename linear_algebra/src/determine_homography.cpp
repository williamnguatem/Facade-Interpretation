//bool LinearAlgebra::DetermineHomography(const Vec2d *pSourcePoints, const Vec2d *pTargetPoints, int nPoints, Mat3d &A, bool bUseSVD)
//	01663 {
//		01664         if (nPoints < 4)
//			01665         {
//				01666                 printf("error: not enough input point pairs for LinearAlgebra::DetermineHomography (must be at least 4)\n");
//				01667                 return false;
//				01668         }
//		01669 
//			01670         // this least squares problem becomes numerically instable when
//			01671         // using float instead of double!!
//			01672         CDoubleMatrix M(8, 2 * nPoints);
//		01673         CDoubleVector b(2 * nPoints);
//		01674         
//			01675         double *data = M.data;
//		01676         
//			01677         for (int i = 0, offset = 0; i < nPoints; i++, offset += 16)
//			01678         {
//				01679                 data[offset] = pSourcePoints[i].x;
//				01680                 data[offset + 1] = pSourcePoints[i].y;
//				01681                 data[offset + 2] = 1;
//				01682                 data[offset + 3] = 0;
//				01683                 data[offset + 4] = 0;
//				01684                 data[offset + 5] = 0;
//				01685                 data[offset + 6] = -pSourcePoints[i].x * pTargetPoints[i].x;
//				01686                 data[offset + 7] = -pSourcePoints[i].y * pTargetPoints[i].x;
//				01687                 
//					01688                 data[offset + 8] = 0;
//				01689                 data[offset + 9] = 0;
//				01690                 data[offset + 10] = 0;
//				01691                 data[offset + 11] = pSourcePoints[i].x;
//				01692                 data[offset + 12] = pSourcePoints[i].y;
//				01693                 data[offset + 13] = 1;
//				01694                 data[offset + 14] = -pSourcePoints[i].x * pTargetPoints[i].y;
//				01695                 data[offset + 15] = -pSourcePoints[i].y * pTargetPoints[i].y;
//				01696 
//					01697                 const int index = 2 * i;
//				01698                 b.data[index] = pTargetPoints[i].x;
//				01699                 b.data[index + 1] = pTargetPoints[i].y;
//				01700         }
//		01701         
//			01702         CDoubleVector x(8);
//		01703         
//			01704         if (bUseSVD)
//			01705                 SolveLinearLeastSquaresSVD(&M, &b, &x);
//		01706         else
//			01707                 SolveLinearLeastSquaresSimple(&M, &b, &x);
//		01708         
//			01709         Math3d::SetMat(A, float(x.data[0]), float(x.data[1]), float(x.data[2]), float(x.data[3]), float(x.data[4]), float(x.data[5]), float(x.data[6]), float(x.data[7]), 1.0f);
//		01710         
//			01711         return true;
//		01712 }