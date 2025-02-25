/*CPP Class for chTimer.h*/

#include <chTimer.h>

class ChTimer {

public:
	ChTimer(){};
	~ChTimer(){};
	
	//
	// Start the Timer
	//
	int 
	start() { chTimerGetTime( &m_start ); return 0; };
	
	//
	// Stop the Timer
	//
	int 
	stop()  { chTimerGetTime( &m_end ); return 0; };
	
	//
	// Get elapsed Time
	//
	double 
	getTime() { return chTimerElapsedTime( &m_start, &m_end ); };
	
	//
	// Get Bandwidth
	//
	double 
	getBandwidth(double size) { return chTimerBandwidth( &m_start, &m_end, size); };

	//
	// Overload + operator to support combining multiple timers
	//
    ChTimer operator+(ChTimer const& obj) {
        ChTimer res;
        res.m_start = m_start;
        res.m_end = m_end;
		res.m_end.tv_sec += (obj.m_end.tv_sec - obj.m_start.tv_sec);
		res.m_end.tv_nsec += (obj.m_end.tv_nsec - obj.m_start.tv_nsec);
        return res;
    }
	

private:
	chTimerTimestamp m_start;
	chTimerTimestamp m_end;

};
