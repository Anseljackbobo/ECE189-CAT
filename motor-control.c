/*
 * ECE 153B - Winter 2020
 *
 * Name(s):
 * Section:
 * Lab: 1A
 */
#include "stm32l476xx.h"
#include "UART.h"
#include "SysClock.h"
#include <stdio.h>
//#include <unistd.h>

void Init(){
	// Enable HSI
	RCC->CR |= RCC_CR_HSION;
	while((RCC->CR & RCC_CR_HSIRDY) == 0);
	
	// Select HSI as system clock source
	RCC->CFGR &= ~RCC_CFGR_SW;
	RCC->CFGR |= RCC_CFGR_SW_HSI;
	while((RCC->CFGR & RCC_CFGR_SWS) == 0);
	
	RCC->AHB2ENR |=RCC_AHB2ENR_GPIOBEN;
	RCC->AHB2ENR |=RCC_AHB2ENR_GPIOEEN;	
	RCC->AHB2ENR |=RCC_AHB2ENR_GPIOAEN;
	
	//RESET ALL MODES
	GPIOB->MODER &= ~GPIO_MODER_MODE2;
	GPIOE->MODER &= ~GPIO_MODER_MODE8;
	
	//SET JS TO INPUT MODE
	GPIOA->MODER &= ~(GPIO_MODER_MODE0
								+GPIO_MODER_MODE1
								+GPIO_MODER_MODE2
								+GPIO_MODER_MODE3
								+GPIO_MODER_MODE5);
	//SET JS TO PILL DOWN						
	GPIOA->PUPDR |= (GPIO_PUPDR_PUPD0_1+
								GPIO_PUPDR_PUPD1_1+
								GPIO_PUPDR_PUPD2_1+
								GPIO_PUPDR_PUPD3_1+
								GPIO_PUPDR_PUPD5_1);
								
								

	//SET THE LEDS TO OUTPUT MODE
	GPIOB->MODER |= GPIO_MODER_MODE2_0;
	GPIOE->MODER |= GPIO_MODER_MODE8_0;
	
		//RESET ALL MODES
	GPIOE->MODER &= ~GPIO_MODER_MODE10;
	GPIOE->MODER &= ~GPIO_MODER_MODE11;

	//SET THE LEDS TO OUTPUT MODE
	GPIOE->MODER |= GPIO_MODER_MODE10_0;
	GPIOE->MODER |= GPIO_MODER_MODE11_0;
	
}

void redOn(){
	GPIOB->ODR |= GPIO_ODR_OD2;
}
void greenOn(){
	GPIOE->ODR |= GPIO_ODR_OD8;
}

void redOff(){
	GPIOB->ODR &= ~GPIO_ODR_OD2;
}

void greenOff(){
	GPIOE->ODR &= ~GPIO_ODR_OD8;
}

void toggleRed(){
	GPIOB->ODR ^= GPIO_ODR_OD2;
}

void toggleGreen(){
	GPIOE->ODR ^= GPIO_ODR_OD8;
}
void Init_USARTx(int part) {
	if(part == 1) {
		UART2_Init();
		UART2_GPIO_Init();
		USART_Init(USART2);
	} else if(part == 2) {
		UART1_Init();
		UART1_GPIO_Init();
		USART_Init(USART1);
	} else {
		// Do nothing...
	}
}
int main()
{

	Init();
  System_Clock_Init();
  Init_USARTx(2);
		char rxByte;
	int flag = 1;


	GPIOE->MODER &= ~GPIO_MODER_MODE10;
	GPIOE->MODER |= GPIO_MODER_MODE10;
	GPIOE->ODR |= GPIO_ODR_OD10;
	USART_TypeDef * USARTx;

/*	
	while(1){
		scanf("%c", &rxByte);

	switch(rxByte){
		case'U': 
			redOn();
		printf("U");
		  break;	
		case'D':
			redOff();
		printf("D");
		  break;
		case'L':
			GPIOE->ODR |= GPIO_ODR_OD10;
		printf("L");
		  break;
		case'R':
			GPIOE->ODR &=~ GPIO_ODR_OD10;
		printf("R");
		  break;
		default:
			flag=0;
	}
	
	if (flag==1){
for(int j=0;j<20;j++)//5.4 degree
{
				  greenOn();
	    for(int i=0;i<150000;i++){}
					greenOff();
			for(int i=0;i<30000;i++){}
}

for(int i=0;i<10;i++)//5.4 degree
{
				  GPIOE->ODR |= GPIO_ODR_OD11;
	    for(int i=0;i<150000;i++){}
					GPIOE->ODR &=~ GPIO_ODR_OD11;
			for(int i=0;i<30000;i++){}
}

    printf("F");
}*/
///*
  RCC->AHB2ENR |=RCC_AHB2ENR_GPIOBEN;
	RCC->AHB2ENR |=RCC_AHB2ENR_GPIOEEN;	

	
	//RESET ALL MODES
	GPIOE->MODER &= ~GPIO_MODER_MODE10;
	GPIOE->MODER &= ~GPIO_MODER_MODE11;

	//SET THE LEDS TO OUTPUT MODE
	GPIOE->MODER |= GPIO_MODER_MODE10_0;
	GPIOE->MODER |= GPIO_MODER_MODE11_0;
for(int j=0;j<50;j++)//5.4 degree
{
				  GPIOE->ODR |= GPIO_ODR_OD11;
	greenOn();
	    for(int i=0;i<1500000;i++){}
					GPIOE->ODR &=~ GPIO_ODR_OD11;
				greenOff();
			for(int i=0;i<300000;i++){}
}
//*/

}
//}