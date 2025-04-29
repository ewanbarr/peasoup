#include <utils/utils.hpp>

/**
 * @brief Parses the angle in sigproc format.
 *
 * This function takes a sigproc angle and parses it into its components.
 *
 * @param sigproc The sigproc angle to be parsed.
 * @param sign The sign of the angle (-1 for negative, 1 for positive).
 * @param first The first component of the angle.
 * @param second The second component of the angle.
 * @param third The third component of the angle.
 */
 void parse_angle_sigproc(double sigproc, int& sign, int& first, int& second, double& third)
 {
 
     sign = sigproc < 0 ? -1 : 1;
     double abs_sigproc = std::abs(sigproc);
 
     first = (int)(abs_sigproc / 1e4);
     second = (int)((abs_sigproc - first * 1e4) / 1e2);
     third = abs_sigproc - first * 1e4 - second * 1e2;
 
 }
  /**
 * Converts a sigproc value to HH:MM:SS format.
 *
 * @param sigproc The sigproc value to convert. This has the format of HHMMSS
 * @param hhmmss  The resulting HH:MM:SS string.
 */
void sigproc_to_hhmmss(double sigproc, std::string& hhmmss)
{ 
    int sign, hh, mm;
    double ss;
    parse_angle_sigproc(sigproc, sign, hh, mm, ss);

    std::stringstream sstream;
    sstream << std::setw(2) << std::setfill('0') << std::fixed << std::setprecision(0) << hh;
    sstream << ":";
    sstream << std::setw(2) << std::setfill('0') << std::fixed << std::setprecision(0) << mm;
    sstream << ":";
    sstream << std::setw(2) << std::setfill('0') << std::fixed << std::setprecision(2) << ss;

    hhmmss = sstream.str();
   
}

/**
 * Converts a signal processing value to a string representation in the format "dd:mm:ss".
 * 
 * @param sigproc The signal processing value to convert.
 * @param ddmmss  The resulting string representation in the format "dd:mm:ss".
 */
void sigproc_to_ddmmss(double sigproc, std::string& ddmmss)
{
    int sign, dd, mm;
    double ss;
    parse_angle_sigproc(sigproc, sign, dd, mm, ss);

    std::stringstream sstream;
    if (sign < 0)
    {
        sstream << std::setw(1) << "-";
    }

    sstream << std::setw(2) << std::setfill('0') << std::fixed << std::setprecision(0) << dd;
    sstream << ":";
    sstream << std::setw(2) << std::setfill('0') << std::fixed << std::setprecision(0) << mm;
    sstream << ":";
    sstream << std::setw(2) << std::setfill('0') << std::fixed << std::setprecision(2) << ss;

    ddmmss = sstream.str();
   
}



