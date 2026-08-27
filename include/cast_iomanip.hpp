#ifndef CAST_IOMANIP_
#define CAST_IOMANIP_

#include <iostream>

namespace cast {


    

/**
* Returns a unique index for output streams. For internal use.
* @return ostream index
*/
inline int get_display_idx() {
    static int iomanip_idx_ = std::ios_base::xalloc();
    return iomanip_idx_;
}

/**
* Sets `output_stream` to display extra information about networks, returning a reference to `output_stream`.
* @param output_stream stream to set
* @return `output_stream` set to verbose
*/
template<typename CharT, typename Traits>
inline std::basic_ostream<CharT, Traits>& verbose(std::basic_ostream<CharT, Traits>& output_stream) {
    output_stream.iword(get_display_idx()) = 1;
    return output_stream;
}

/**
* Stops `output_stream` from printing detailed information about networks, returning a reference to `output_stream`.
* @param output_stream stream to set
* @return `output_stream` set to no-verbose
*/
template<typename CharT, typename Traits>
inline std::basic_ostream<CharT, Traits>& noverbose(std::basic_ostream<CharT, Traits>& output_stream) {
    output_stream.iword(get_display_idx()) = 0;
    return output_stream;
}




}
#endif